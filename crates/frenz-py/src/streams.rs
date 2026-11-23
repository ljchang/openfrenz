//! Per-stream sample buffers, filled from the band's events and read from Python.
//!
//! A task on the runtime turns every decoded packet into rows in a ring buffer per stream. Python
//! never takes part in that, so a slow reader cannot delay Bluetooth, decoding or recording; it
//! can only fall behind its own buffer, which is counted and reported.

use std::collections::VecDeque;
use std::sync::{Mutex, PoisonError};

use frenz::Event;
use frenz::clock::anchor_wall_ns;
use frenz::proto::{Characteristic, EegPacket, MotionPacket, Packet, ParseError, PpgPacket};
use frenz_export::timing::{ClockFit, Segmenter, fit_lower_envelope, fit_nominal};
use tokio::sync::broadcast::{Receiver, error::RecvError};
use tokio::sync::watch;

/// Most values a packet sample carries (the six EEG slots).
const WIDE: usize = 6;
/// Most channels a stream delivers.
const CHANNELS: usize = 4;
/// Packets kept for the live clock fit of one segment.
const FIT_WINDOW: usize = 1024;
/// Rows kept for streams without a steady rate (heart rate, SpO2, battery, contact) per second
/// of buffer; these arrive at most about once a second.
const SLOW_ROWS_PER_SECOND: f64 = 2.0;

/// One sample as Python receives it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Row {
    /// Estimated acquisition time (Unix seconds).
    pub(crate) t: f64,
    /// Arrival time of the packet that carried the sample (Unix seconds).
    pub(crate) arrival: f64,
    /// The band's clock for the packet (whole seconds).
    pub(crate) device_time: i64,
    /// Uninterrupted stretch the sample belongs to.
    pub(crate) segment: i64,
    /// Position of the sample within its segment, counting samples that were lost.
    pub(crate) index: i64,
    /// The packet repeated the previous packet's samples.
    pub(crate) duplicate: bool,
    /// Every sample in the packet was identical (frozen sensor output).
    pub(crate) held: bool,
    /// Channel values, in the stream's channel order.
    pub(crate) values: [i32; CHANNELS],
}

/// Samples handed to Python in one call.
#[derive(Debug)]
pub(crate) struct Chunk {
    /// Stream name.
    pub(crate) stream: &'static str,
    /// Channel names.
    pub(crate) channels: &'static [&'static str],
    /// Whether the stream has a steady sample rate (and so segments and repeat flags).
    pub(crate) sampled: bool,
    /// The samples, oldest first.
    pub(crate) rows: Vec<Row>,
    /// Samples that left the buffer unread before the first of these.
    pub(crate) dropped: u64,
}

/// Why no more data will come.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum End {
    /// The replay finished or the band was closed.
    Finished,
    /// The link was lost and is not coming back.
    Lost,
}

struct Spec {
    name: &'static str,
    channels: &'static [&'static str],
    /// Columns of a packet's samples that become the channels.
    pick: &'static [usize],
    rate_hz: Option<f64>,
}

const SPECS: [Spec; 8] = [
    Spec {
        name: "eeg",
        channels: &["forehead_right", "comb_left", "forehead_left", "comb_right"],
        pick: &[0, 1, 3, 4],
        rate_hz: Some(EegPacket::RATE_HZ),
    },
    Spec {
        name: "accel",
        channels: &["x", "y", "z"],
        pick: &[0, 1, 2],
        rate_hz: Some(MotionPacket::RATE_HZ),
    },
    Spec {
        name: "gyro",
        channels: &["x", "y", "z"],
        pick: &[0, 1, 2],
        rate_hz: Some(MotionPacket::RATE_HZ),
    },
    Spec {
        name: "ppg",
        channels: &["tag2", "tag3"],
        pick: &[1, 2],
        rate_hz: Some(PpgPacket::RATE_HZ),
    },
    Spec {
        name: "heart_rate",
        channels: &["value", "check"],
        pick: &[0, 1],
        rate_hz: None,
    },
    Spec {
        name: "spo2",
        channels: &["value", "check"],
        pick: &[0, 1],
        rate_hz: None,
    },
    Spec {
        name: "battery",
        channels: &["percent"],
        pick: &[0],
        rate_hz: None,
    },
    Spec {
        name: "contact",
        channels: &["state"],
        pick: &[0],
        rate_hz: None,
    },
];

const EEG: usize = 0;
const ACCEL: usize = 1;
const GYRO: usize = 2;
const PPG: usize = 3;
const HEART_RATE: usize = 4;
const SPO2: usize = 5;
const BATTERY: usize = 6;
const CONTACT: usize = 7;

/// Stream names, in a fixed order.
pub(crate) fn names() -> impl Iterator<Item = &'static str> {
    SPECS.iter().map(|s| s.name)
}

/// Index of the stream called `name`.
pub(crate) fn index(name: &str) -> Option<usize> {
    SPECS.iter().position(|s| s.name == name)
}

/// Counters for one stream, for `health()`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StreamHealth {
    pub(crate) name: &'static str,
    /// Samples received since the connection opened.
    pub(crate) samples: u64,
    /// Samples currently in the buffer.
    pub(crate) buffered: usize,
    /// Samples that left the buffer before `read()` took them.
    pub(crate) dropped: u64,
    /// Samples with a non-zero value in a column that is normally zero and is not delivered.
    pub(crate) inactive_nonzero: u64,
    /// Segment of the newest sample.
    pub(crate) segment: i64,
    /// Live clock fit of the current segment, for streams with a steady rate.
    pub(crate) fit: Option<ClockFit>,
}

struct Stream {
    spec: &'static Spec,
    capacity: usize,
    rows: VecDeque<Row>,
    /// Samples ever pushed.
    total: u64,
    /// Position of the next sample `read()` returns.
    cursor: u64,
    dropped: u64,
    inactive_nonzero: u64,
    segmenter: Option<Segmenter>,
    segment: i64,
    /// `(index of the packet's last sample, arrival)` for the current segment.
    points: VecDeque<(f64, f64)>,
    fit: Option<ClockFit>,
    last_t: f64,
    previous: Vec<[i32; WIDE]>,
    force_break: bool,
    after_start: bool,
}

impl Stream {
    fn new(spec: &'static Spec, buffer_seconds: f64) -> Self {
        let per_second = spec.rate_hz.unwrap_or(SLOW_ROWS_PER_SECOND);
        let capacity = ((buffer_seconds * per_second).ceil() as usize).max(64);
        Self {
            spec,
            capacity,
            rows: VecDeque::new(),
            total: 0,
            cursor: 0,
            dropped: 0,
            inactive_nonzero: 0,
            segmenter: spec.rate_hz.map(Segmenter::new),
            segment: 0,
            points: VecDeque::new(),
            fit: None,
            last_t: f64::NEG_INFINITY,
            previous: Vec::new(),
            force_break: false,
            after_start: false,
        }
    }

    fn values(&self, wide: &[i32; WIDE]) -> [i32; CHANNELS] {
        let mut values = [0; CHANNELS];
        for (value, column) in values.iter_mut().zip(self.spec.pick) {
            *value = wide.get(*column).copied().unwrap_or(0);
        }
        values
    }

    fn push_row(&mut self, row: Row) {
        if self.rows.len() == self.capacity {
            self.rows.pop_front();
        }
        self.rows.push_back(row);
        self.total += 1;
    }

    /// Adds one packet. `width` is how many of each sample's values the packet really carries.
    fn push(&mut self, arrival: f64, device_time: u32, sequence: Option<u8>, samples: &[[i32; WIDE]], width: usize) {
        let pick = self.spec.pick;
        let unused = (0..width).filter(|c| !pick.contains(c));
        self.inactive_nonzero += samples
            .iter()
            .filter(|s| unused.clone().any(|c| s.get(c).is_some_and(|v| *v != 0)))
            .count() as u64;
        let device_time = i64::from(device_time);
        let (Some(segmenter), Some(rate_hz)) = (self.segmenter.as_mut(), self.spec.rate_hz) else {
            // No steady rate: the arrival time is the best estimate, as in exports.
            for sample in samples {
                let values = self.values(sample);
                self.push_row(Row {
                    t: arrival,
                    arrival,
                    device_time,
                    segment: 0,
                    index: 0,
                    duplicate: false,
                    held: false,
                    values,
                });
            }
            return;
        };
        let placed = segmenter.push(
            arrival,
            device_time as u32,
            sequence,
            samples.len(),
            std::mem::take(&mut self.force_break),
            std::mem::take(&mut self.after_start),
        );
        if placed.segment != self.segment {
            self.segment = placed.segment;
            self.points.clear();
            self.fit = None;
            self.last_t = f64::NEG_INFINITY;
        }
        let duplicate = !self.previous.is_empty() && self.previous == samples;
        let held = samples.len() > 1 && samples.windows(2).all(|w| w.first() == w.get(1));
        self.previous.clear();
        self.previous.extend_from_slice(samples);
        // Repeated and frozen packets say nothing about the clock, so they are left out of the fit.
        if !duplicate && !held {
            if self.points.len() == FIT_WINDOW {
                self.points.pop_front();
            }
            let last = placed.first_sample + samples.len() as i64 - 1;
            self.points.push_back((last as f64, arrival));
            let points = self.points.make_contiguous();
            let fit = fit_lower_envelope(points, rate_hz);
            // Early in a segment a few late packets can tilt the fit. A rate more than 5 % from
            // nominal is not believable, so until the data says otherwise only the offset is fitted.
            let believable = (fit.period * rate_hz - 1.0).abs() < 0.05;
            self.fit = Some(if believable { fit } else { fit_nominal(points, rate_hz) });
        }
        let period = self.fit.map_or(1.0 / rate_hz, |f| f.period);
        for (k, sample) in samples.iter().enumerate() {
            let n = (placed.first_sample + k as i64) as f64;
            let estimate = self.fit.map_or(arrival, |f| f.time(n));
            // The fit improves as packets arrive and may move earlier. Time never runs backwards
            // within a segment: a correction is taken up over the following samples instead.
            let t = estimate.max(self.last_t + 0.5 * period);
            self.last_t = t;
            let values = self.values(sample);
            self.push_row(Row {
                t,
                arrival,
                device_time,
                segment: placed.segment,
                index: placed.first_sample + k as i64,
                duplicate,
                held,
                values,
            });
        }
    }

    fn oldest(&self) -> u64 {
        self.total - self.rows.len() as u64
    }

    fn chunk(&self, rows: Vec<Row>, dropped: u64) -> Chunk {
        Chunk {
            stream: self.spec.name,
            channels: self.spec.channels,
            sampled: self.spec.rate_hz.is_some(),
            rows,
            dropped,
        }
    }

    /// Everything not yet read, or `None` if there is nothing new.
    fn take(&mut self) -> Option<Chunk> {
        if self.cursor == self.total {
            return None;
        }
        let oldest = self.oldest();
        let dropped = oldest.saturating_sub(self.cursor);
        self.dropped += dropped;
        let skip = self.cursor.saturating_sub(oldest) as usize;
        let rows = self.rows.iter().skip(skip).copied().collect();
        self.cursor = self.total;
        Some(self.chunk(rows, dropped))
    }

    /// The most recent `seconds` of data, without consuming it.
    fn latest(&self, seconds: f64) -> Chunk {
        let count = match self.spec.rate_hz {
            Some(rate) => (seconds * rate).round() as usize,
            None => {
                let newest = self.rows.back().map_or(0.0, |r| r.arrival);
                self.rows
                    .iter()
                    .rev()
                    .take_while(|r| newest - r.arrival < seconds)
                    .count()
            }
        };
        let skip = self.rows.len().saturating_sub(count);
        self.chunk(self.rows.iter().skip(skip).copied().collect(), 0)
    }
}

/// Most events kept for Python to read; older ones are dropped and counted.
const NOTE_CAPACITY: usize = 4096;

/// A value attached to an event.
#[derive(Debug, Clone)]
pub(crate) enum Detail {
    /// A count or number.
    Int(u64),
    /// A name or message.
    Text(String),
}

/// Something that happened on the connection, as Python receives it.
#[derive(Debug, Clone)]
pub(crate) struct Note {
    /// What happened, e.g. `"sequence_gap"`.
    pub(crate) kind: &'static str,
    /// When (Unix seconds).
    pub(crate) t: f64,
    /// Further facts, by name.
    pub(crate) detail: Vec<(&'static str, Detail)>,
}

fn unix(mono_ns: i64) -> f64 {
    // One fixed mapping from the monotonic clock to Unix time, so times never step.
    (mono_ns + anchor_wall_ns()) as f64 / 1e9
}

/// The event Python is told about, if any. Packets are data, not events.
fn note_for(event: &Event) -> Option<Note> {
    let note = |kind, mono_ns: &i64, detail| {
        Some(Note {
            kind,
            t: unix(*mono_ns),
            detail,
        })
    };
    match event {
        Event::Packet { .. } => None,
        Event::DecodeError {
            mono_ns,
            characteristic,
            error,
        } => note(
            "decode_error",
            mono_ns,
            vec![
                ("stream", Detail::Text(characteristic.name().to_owned())),
                ("error", Detail::Text(error.to_string())),
            ],
        ),
        Event::SequenceGap {
            mono_ns,
            expected,
            received,
        } => note(
            "sequence_gap",
            mono_ns,
            vec![
                ("expected", Detail::Int(u64::from(*expected))),
                ("received", Detail::Int(u64::from(*received))),
                ("frames_lost", Detail::Int(u64::from(received.wrapping_sub(*expected)))),
            ],
        ),
        Event::CounterRestart { mono_ns } => note("counter_restart", mono_ns, vec![]),
        Event::OffHead { mono_ns } => note("off_head", mono_ns, vec![]),
        Event::OnHead { mono_ns } => note("on_head", mono_ns, vec![]),
        Event::PowerOffWarning { mono_ns } => note("power_off_warning", mono_ns, vec![]),
        Event::Disconnected { mono_ns } => note("disconnected", mono_ns, vec![]),
        Event::Reconnecting { mono_ns, attempt } => note(
            "reconnecting",
            mono_ns,
            vec![("attempt", Detail::Int(u64::from(*attempt)))],
        ),
        Event::Reconnected { mono_ns } => note("reconnected", mono_ns, vec![]),
        Event::ReconnectFailed { mono_ns } => note("reconnect_failed", mono_ns, vec![]),
    }
}

struct Inner {
    streams: Vec<Stream>,
    ended: Option<End>,
    events_missed: u64,
    notes: VecDeque<Note>,
    notes_dropped: u64,
}

impl Inner {
    fn note(&mut self, note: Note) {
        if self.notes.len() == NOTE_CAPACITY {
            self.notes.pop_front();
            self.notes_dropped += 1;
        }
        self.notes.push_back(note);
    }
}

/// The buffers of one connection.
pub(crate) struct Streams {
    inner: Mutex<Inner>,
    /// Bumped whenever data arrives or the connection ends, to wake waiting readers.
    changed: watch::Sender<u64>,
}

impl Streams {
    pub(crate) fn new(buffer_seconds: f64) -> Self {
        Self {
            inner: Mutex::new(Inner {
                streams: SPECS.iter().map(|spec| Stream::new(spec, buffer_seconds)).collect(),
                ended: None,
                events_missed: 0,
                notes: VecDeque::new(),
                notes_dropped: 0,
            }),
            changed: watch::Sender::new(0),
        }
    }

    fn with<T>(&self, f: impl FnOnce(&mut Inner) -> T) -> T {
        f(&mut self.inner.lock().unwrap_or_else(PoisonError::into_inner))
    }

    fn wake(&self) {
        self.changed
            .send_modify(|generation| *generation = generation.wrapping_add(1));
    }

    /// Marks the connection as over and wakes every waiting reader.
    pub(crate) fn end(&self, end: End) {
        self.with(|inner| inner.ended = Some(end));
        self.wake();
    }

    /// Why the connection is over, if it is.
    pub(crate) fn ended(&self) -> Option<End> {
        self.with(|inner| inner.ended)
    }

    /// Events the buffers never saw because this module fell behind the band's event queue.
    pub(crate) fn events_missed(&self) -> u64 {
        self.with(|inner| inner.events_missed)
    }

    /// Events that left the queue before Python read them.
    pub(crate) fn events_dropped(&self) -> u64 {
        self.with(|inner| inner.notes_dropped)
    }

    /// Waits for the next event. `Err` means the connection is over and every event was read.
    pub(crate) async fn next_note(&self) -> Result<Note, End> {
        let mut changed = self.changed.subscribe();
        loop {
            let step = self.with(|inner| match (inner.notes.pop_front(), inner.ended) {
                (Some(note), _) => Some(Ok(note)),
                (None, Some(end)) => Some(Err(end)),
                (None, None) => None,
            });
            if let Some(result) = step {
                return result;
            }
            if changed.changed().await.is_err() {
                return Err(End::Finished);
            }
        }
    }

    pub(crate) fn health(&self) -> Vec<StreamHealth> {
        self.with(|inner| {
            inner
                .streams
                .iter()
                .map(|s| StreamHealth {
                    name: s.spec.name,
                    samples: s.total,
                    buffered: s.rows.len(),
                    dropped: s.dropped,
                    inactive_nonzero: s.inactive_nonzero,
                    segment: s.segment,
                    fit: s.fit,
                })
                .collect()
        })
    }

    /// Waits for unread samples on stream `index`. `Err` means the connection is over and
    /// everything has been read.
    pub(crate) async fn read(&self, index: usize) -> Result<Chunk, End> {
        let mut changed = self.changed.subscribe();
        loop {
            let step = self.with(|inner| {
                let taken = inner.streams.get_mut(index).and_then(Stream::take);
                if let Some(chunk) = taken.as_ref().filter(|chunk| chunk.dropped > 0) {
                    // The reader was too slow for its buffer: say so as an event too.
                    inner.note(Note {
                        kind: "lagged",
                        t: unix(frenz::clock::mono_ns()),
                        detail: vec![
                            ("stream", Detail::Text(chunk.stream.to_owned())),
                            ("dropped", Detail::Int(chunk.dropped)),
                        ],
                    });
                }
                match (taken, inner.ended) {
                    (Some(chunk), _) => Some(Ok(chunk)),
                    (None, Some(end)) => Some(Err(end)),
                    (None, None) => None,
                }
            });
            if let Some(result) = step {
                return result;
            }
            if changed.changed().await.is_err() {
                return Err(End::Finished);
            }
        }
    }

    /// The most recent `seconds` of stream `index`, without consuming it.
    pub(crate) fn latest(&self, index: usize, seconds: f64) -> Option<Chunk> {
        self.with(|inner| inner.streams.get(index).map(|s| s.latest(seconds)))
    }

    fn push(
        &self,
        index: usize,
        mono_ns: i64,
        device_time: u32,
        sequence: Option<u8>,
        samples: &[[i32; WIDE]],
        width: usize,
    ) {
        let arrival = unix(mono_ns);
        self.with(|inner| {
            if let Some(stream) = inner.streams.get_mut(index) {
                stream.push(arrival, device_time, sequence, samples, width);
            }
        });
    }

    fn handle(&self, event: &Event, reconnects: bool, end: End) {
        if let Some(note) = note_for(event) {
            self.with(|inner| inner.note(note));
        }
        match event {
            Event::Packet { mono_ns, packet } => {
                let wide = |values: &[i32]| {
                    let mut out = [0; WIDE];
                    for (o, v) in out.iter_mut().zip(values) {
                        *o = *v;
                    }
                    out
                };
                match packet.as_ref() {
                    Packet::Eeg(e) => self.push(EEG, *mono_ns, e.device_time, Some(e.sequence), &e.samples, 6),
                    Packet::Accelerometer(m) | Packet::Gyroscope(m) => {
                        let index = if matches!(packet.as_ref(), Packet::Accelerometer(_)) {
                            ACCEL
                        } else {
                            GYRO
                        };
                        let samples: Vec<_> = m.samples.iter().map(|s| wide(&s.map(i32::from))).collect();
                        self.push(index, *mono_ns, m.device_time, None, &samples, 3);
                    }
                    Packet::Ppg(p) => {
                        // 19-bit values, so they fit a signed 32-bit integer.
                        let samples: Vec<_> = p.samples().iter().map(|s| wide(&s.map(|v| v as i32))).collect();
                        self.push(PPG, *mono_ns, p.device_time, None, &samples, 3);
                    }
                    Packet::HeartRate(v) | Packet::Spo2(v) => {
                        let index = if matches!(packet.as_ref(), Packet::HeartRate(_)) {
                            HEART_RATE
                        } else {
                            SPO2
                        };
                        let sample = wide(&[i32::from(v.value), i32::from(v.check)]);
                        self.push(index, *mono_ns, v.device_time, None, &[sample], 2);
                    }
                    Packet::Battery(b) => {
                        self.push(
                            BATTERY,
                            *mono_ns,
                            b.device_time,
                            None,
                            &[wide(&[i32::from(b.percent)])],
                            1,
                        );
                    }
                    Packet::Contact(c) => {
                        self.push(
                            CONTACT,
                            *mono_ns,
                            c.device_time,
                            None,
                            &[wide(&[i32::from(c.state)])],
                            1,
                        );
                    }
                    _ => {}
                }
            }
            // An EEG frame that could not be read: what follows is a new segment, as in exports.
            Event::DecodeError {
                characteristic: Characteristic::Eeg,
                error: ParseError::EegFrameType(_) | ParseError::EegUnsequenced,
                ..
            } => self.with(|inner| {
                if let Some(eeg) = inner.streams.get_mut(EEG) {
                    eeg.force_break = true;
                }
            }),
            Event::CounterRestart { .. } => self.with(|inner| {
                if let Some(eeg) = inner.streams.get_mut(EEG) {
                    eeg.after_start = true;
                }
            }),
            Event::Disconnected { .. } if !reconnects => self.end(end),
            Event::ReconnectFailed { .. } => self.end(end),
            _ => {}
        }
    }

    fn missed(&self, events: u64) {
        self.with(|inner| {
            inner.events_missed += events;
            inner.note(Note {
                kind: "events_missed",
                t: unix(frenz::clock::mono_ns()),
                detail: vec![("count", Detail::Int(events))],
            });
            // Packets are missing from the buffers (not from the recording), so nothing after
            // this point may be joined to what came before.
            for stream in &mut inner.streams {
                stream.force_break = true;
            }
        });
    }
}

/// Feeds `streams` from the band's events until the band goes away. `reconnects` says whether a
/// lost link is retried; `end` is what a final disconnect means for this connection.
pub(crate) async fn pump(streams: std::sync::Arc<Streams>, mut events: Receiver<Event>, reconnects: bool, end: End) {
    loop {
        match events.recv().await {
            Ok(event) => {
                streams.handle(&event, reconnects, end);
                streams.wake();
            }
            Err(RecvError::Lagged(missed)) => streams.missed(missed),
            Err(RecvError::Closed) => {
                streams.end(end);
                return;
            }
        }
    }
}
