//! Running a model on a recording and on a band.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use frenz::Band;
use frenz_export::Session;
use frenz_export::table::{Column, Table};
use tokio::task::JoinHandle;

use crate::Error;
use crate::model::{Model, Output, Take};
use crate::stepper::Stepper;
use crate::streams::{self, Chunk, End, Streams};

fn numbers(column: &Column) -> Option<Vec<f64>> {
    Some(match column {
        Column::F64(v) => v.clone(),
        Column::I64(v) => v.iter().map(|x| *x as f64).collect(),
        Column::I32(v) => v.iter().map(|x| f64::from(*x)).collect(),
        Column::I16(v) => v.iter().map(|x| f64::from(*x)).collect(),
        Column::I8(v) => v.iter().map(|x| f64::from(*x)).collect(),
        Column::U32(v) => v.iter().map(|x| f64::from(*x)).collect(),
        Column::U8(v) => v.iter().map(|x| f64::from(*x)).collect(),
        _ => return None,
    })
}

fn column<'a>(table: &'a Table, name: &str) -> Option<&'a Column> {
    table.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
}

/// One stream of a session, as the stepper takes it.
struct Signal {
    t: Vec<f64>,
    values: Vec<Vec<f64>>,
    segment: Vec<i64>,
}

/// The rows of `stream` in `session`, in time order, without repeated or frozen packets.
/// `None` when the session does not hold the stream.
fn signal(session: &Session, stream: &str) -> Result<Option<Signal>, Error> {
    let channels =
        streams::channels(stream).ok_or_else(|| Error::Spec(format!("{stream} is not a stream the band sends")))?;
    // Heart rate and SpO2 share one table, told apart by its `kind` column.
    let vital = matches!(stream, "heart_rate" | "spo2");
    let Some(table) = session.tables.get(if vital { "vitals" } else { stream }) else {
        return Ok(None);
    };
    let bad = |what: &str| Error::Files(format!("the {stream} table has no usable {what} column"));
    let t = column(table, "t").and_then(numbers).ok_or_else(|| bad("t"))?;
    let mut keep = vec![true; t.len()];
    for flag in ["duplicate", "held"] {
        if let Some(Column::Bool(flags)) = column(table, flag) {
            for (k, f) in keep.iter_mut().zip(flags) {
                *k &= !f;
            }
        }
    }
    if vital {
        let Some(Column::Str(kinds)) = column(table, "kind") else {
            return Err(bad("kind"));
        };
        for (k, kind) in keep.iter_mut().zip(kinds) {
            *k &= kind == stream;
        }
    }
    let mut order: Vec<usize> = keep.iter().enumerate().filter(|(_, k)| **k).map(|(i, _)| i).collect();
    let time = |i: &usize| t.get(*i).copied().unwrap_or(f64::NAN);
    order.sort_by(|a, b| time(a).total_cmp(&time(b)));
    if order.is_empty() {
        return Ok(None);
    }
    let pick = |all: &[f64]| {
        order
            .iter()
            .map(|i| all.get(*i).copied().unwrap_or(f64::NAN))
            .collect::<Vec<_>>()
    };
    let mut values = Vec::new();
    for name in channels {
        values.push(pick(&column(table, name).and_then(numbers).ok_or_else(|| bad(name))?));
    }
    let segment = match column(table, "segment") {
        Some(Column::I64(all)) => order.iter().map(|i| all.get(*i).copied().unwrap_or(0)).collect(),
        _ => vec![0; order.len()],
    };
    Ok(Some(Signal {
        t: pick(&t),
        values,
        segment,
    }))
}

/// What a model gave over a whole recording.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    /// One output per step that produced one, in time order.
    pub outputs: Vec<Output>,
    /// Steps skipped because data was missing.
    pub skipped: u64,
    /// The skipped steps, counted by the stream that was missing from each.
    pub missed: BTreeMap<String, u64>,
}

/// Runs `model` over a session read with [`frenz_export::read`].
pub fn run_session<M: Model>(model: M, session: &Session) -> Result<Run, Error> {
    let inputs = model.spec().inputs.clone();
    let mut stepper = Stepper::new(model)?;
    // The first stream last, so its steps find the others complete.
    for input in inputs.iter().rev() {
        match signal(session, &input.stream)? {
            Some(s) => stepper.push(&input.stream, &s.t, &s.values, Some(&s.segment), false),
            None if input.required && input.take != Take::Arrived => {
                return Err(Error::MissingStream(input.stream.clone()));
            }
            None => {}
        }
    }
    let outputs = stepper.pop(true)?;
    Ok(Run {
        outputs,
        skipped: stepper.skipped(),
        missed: stepper.missed().clone(),
    })
}

/// Runs `model` over a `.sqlite` recording.
pub fn run_recording<M: Model>(model: M, recording: &Path) -> Result<Run, Error> {
    run_session(model, &frenz_export::read(recording)?)
}

/// Seconds of samples kept per stream while the model is busy.
const BUFFER_SECONDS: f64 = 60.0;

/// A model running on a connected or replayed band.
///
/// Dropping it stops reading; the band itself is left as it was.
pub struct Live<M: Model> {
    stepper: Stepper<M>,
    streams: Arc<Streams>,
    /// Buffer index of each input, in the order of the model's inputs.
    indexes: Vec<(String, usize)>,
    pump: JoinHandle<()>,
    ready: VecDeque<Output>,
    ended: bool,
}

impl<M: Model> Live<M> {
    /// Starts feeding `model` from `band`. Call it before anything else subscribes to a replayed
    /// band, so the model sees the recording from its start. `reconnects` says whether the band
    /// retries a lost link (false for a replay): if so a disconnect is not the end.
    ///
    /// Pauses and lost data are recognised by when packets arrive, so a replay much faster than
    /// the recording hides them. To score a recording quickly use [`run_recording`].
    pub fn new(model: M, band: &Band, reconnects: bool) -> Result<Self, Error> {
        let mut indexes = Vec::new();
        for input in &model.spec().inputs {
            let index = streams::index(&input.stream)
                .ok_or_else(|| Error::Spec(format!("{} is not a stream the band sends", input.stream)))?;
            indexes.push((input.stream.clone(), index));
        }
        let stepper = Stepper::new(model)?;
        let streams = Arc::new(Streams::new(BUFFER_SECONDS));
        let pump = tokio::spawn(streams::pump(
            streams.clone(),
            band.subscribe(),
            reconnects,
            End::Finished,
        ));
        Ok(Self {
            stepper,
            streams,
            indexes,
            pump,
            ready: VecDeque::new(),
            ended: false,
        })
    }

    /// The model.
    pub fn model(&self) -> &M {
        self.stepper.model()
    }

    /// Steps skipped so far because data was missing. A model that gives no output while this
    /// number grows is waiting for a stream it requires: [`Live::missed`] says which.
    pub fn skipped(&self) -> u64 {
        self.stepper.skipped()
    }

    /// The skipped steps, counted by the stream that was missing from each.
    pub fn missed(&self) -> &BTreeMap<String, u64> {
        self.stepper.missed()
    }

    fn push(&mut self, stream: &str, chunk: &Chunk) {
        let rows: Vec<_> = chunk.rows.iter().filter(|r| !(r.duplicate || r.held)).collect();
        let t: Vec<f64> = rows.iter().map(|r| r.t).collect();
        let segment: Vec<i64> = rows.iter().map(|r| r.segment).collect();
        let values: Vec<Vec<f64>> = (0..chunk.channels.len())
            .map(|c| {
                rows.iter()
                    .map(|r| r.values.get(c).copied().map_or(f64::NAN, f64::from))
                    .collect()
            })
            .collect();
        self.stepper
            .push(stream, &t, &values, Some(&segment), chunk.dropped > 0);
    }

    /// Takes whatever the other streams have, without waiting.
    fn drain_others(&mut self) {
        for (stream, index) in self.indexes.clone().into_iter().skip(1) {
            if let Some(Ok(chunk)) = self.streams.try_read(index) {
                self.push(&stream, &chunk);
            }
        }
    }

    /// Waits for the model's next output. `None` once the band has ended and every complete step
    /// has been returned.
    pub async fn next(&mut self) -> Result<Option<Output>, Error> {
        loop {
            if let Some(output) = self.ready.pop_front() {
                return Ok(Some(output));
            }
            if self.ended {
                return Ok(None);
            }
            let Some((clock, index)) = self.indexes.first().cloned() else {
                return Ok(None);
            };
            match self.streams.read(index).await {
                Ok(chunk) => self.push(&clock, &chunk),
                Err(_) => self.ended = true,
            }
            // After the first stream, so what arrived while waiting for it is included.
            self.drain_others();
            self.ready.extend(self.stepper.pop(self.ended)?);
        }
    }
}

impl<M: Model> Drop for Live<M> {
    fn drop(&mut self) {
        self.pump.abort();
    }
}
