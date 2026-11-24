//! Cutting streams into steps.

use crate::Error;
use crate::model::{Block, Input, Model, Output, Step, Take};

#[derive(Default)]
struct Buffer {
    t: Vec<f64>,
    /// `x[channel][sample]`.
    x: Vec<Vec<f64>>,
}

impl Buffer {
    fn extend(&mut self, t: &[f64], values: &[Vec<f64>]) {
        if self.x.is_empty() {
            self.x = vec![Vec::new(); values.len()];
        }
        self.t.extend_from_slice(t);
        for (mine, theirs) in self.x.iter_mut().zip(values) {
            mine.extend_from_slice(theirs);
        }
    }

    fn drop_front(&mut self, n: usize) {
        let n = n.min(self.t.len());
        self.t.drain(..n);
        for channel in &mut self.x {
            channel.drain(..n.min(channel.len()));
        }
    }

    fn block(&self, from: usize, to: usize) -> Block {
        Block {
            t: self.t.get(from..to).unwrap_or_default().to_vec(),
            values: self
                .x
                .iter()
                .map(|c| c.get(from..to).unwrap_or_default().to_vec())
                .collect(),
        }
    }

    /// Time spanned by samples `from..to`.
    fn span(&self, from: usize, to: usize) -> f64 {
        match (self.t.get(from), to.checked_sub(1).and_then(|i| self.t.get(i))) {
            (Some(a), Some(b)) => b - a,
            _ => 0.0,
        }
    }
}

/// Cuts streams into steps for one model.
///
/// [`Stepper::push`] samples as they come and [`Stepper::pop`] the outputs that have become
/// available. The model is reset at the start, wherever the first stream starts a new segment,
/// and after every step that had to be skipped.
pub struct Stepper<M: Model> {
    model: M,
    inputs: Vec<Input>,
    /// Samples of the first stream per step.
    size: usize,
    max_step_seconds: f64,
    outputs: usize,
    /// One buffer per input, in the order of `inputs`.
    buffers: Vec<Buffer>,
    segment: Option<i64>,
    /// Positions in the first stream's buffer where a new segment starts.
    breaks: Vec<usize>,
    /// End of the previous step when the next one follows it directly.
    previous_end: Option<f64>,
    skipped: u64,
}

impl<M: Model> Stepper<M> {
    /// Checks the model's spec and resets the model.
    pub fn new(mut model: M) -> Result<Self, Error> {
        let spec = model.spec().clone();
        let size = match spec.inputs.first() {
            Some(Input {
                take: Take::Samples(n),
                required: true,
                ..
            }) if *n > 0 => *n,
            _ => {
                return Err(Error::Spec(
                    "the first input sets the steps: it must be required and take a fixed number of samples".into(),
                ));
            }
        };
        for (i, input) in spec.inputs.iter().enumerate() {
            if spec.inputs.iter().take(i).any(|other| other.stream == input.stream) {
                return Err(Error::Spec(format!("{} is listed twice", input.stream)));
            }
            if input.take == Take::Samples(0) {
                return Err(Error::Spec(format!("{} takes no samples", input.stream)));
            }
        }
        if spec.max_step_seconds.is_nan() || spec.max_step_seconds <= 0.0 {
            return Err(Error::Spec("max_step_seconds must be positive".into()));
        }
        model.reset();
        Ok(Self {
            model,
            buffers: spec.inputs.iter().map(|_| Buffer::default()).collect(),
            inputs: spec.inputs,
            size,
            max_step_seconds: spec.max_step_seconds,
            outputs: spec.outputs.len(),
            segment: None,
            breaks: Vec::new(),
            previous_end: None,
            skipped: 0,
        })
    }

    /// The model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Steps skipped so far because data was missing.
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// Adds samples of `stream`, oldest first: their times, `values[channel][sample]`, and for
    /// the first stream the segment of each sample. `gap` says samples were lost just before
    /// these. Streams the model does not read are ignored.
    pub fn push(&mut self, stream: &str, t: &[f64], values: &[Vec<f64>], segment: Option<&[i64]>, gap: bool) {
        let Some(index) = self.inputs.iter().position(|i| i.stream == stream) else {
            return;
        };
        let Some(buffer) = self.buffers.get_mut(index) else {
            return;
        };
        if t.is_empty() {
            return;
        }
        if index == 0 {
            let held = buffer.t.len();
            let mut previous = if gap { None } else { self.segment };
            let mut first = self.segment.is_none() && !gap;
            for k in 0..t.len() {
                let now = segment.and_then(|s| s.get(k)).copied().unwrap_or(0);
                if !first && previous != Some(now) {
                    self.breaks.push(held + k);
                }
                first = false;
                previous = Some(now);
            }
            self.segment = previous;
        }
        buffer.extend(t, values);
    }

    fn restart(&mut self) {
        self.model.reset();
        self.previous_end = None;
    }

    fn drop_clock(&mut self, n: usize) {
        if let Some(clock) = self.buffers.first_mut() {
            clock.drop_front(n);
        }
        self.breaks = self.breaks.iter().filter(|b| **b >= n).map(|b| b - n).collect();
    }

    /// Forgets data of the other streams that no later step can use.
    fn trim(&mut self, before: f64) {
        for buffer in self.buffers.iter_mut().skip(1) {
            let keep = buffer.t.partition_point(|t| *t < before);
            buffer.drop_front(keep);
        }
    }

    /// Outputs of every step that is complete. `end` says no more samples will come, so steps
    /// still waiting for another stream are no longer held back.
    pub fn pop(&mut self, end: bool) -> Result<Vec<Output>, Error> {
        let mut out = Vec::new();
        let n = self.size;
        loop {
            if let Some(b) = self.breaks.first().copied().filter(|b| *b < n) {
                // A new segment starts inside this step: begin again there.
                self.breaks.remove(0);
                self.drop_clock(b);
                self.restart();
                continue;
            }
            let Some(clock) = self.buffers.first() else { break };
            let held = clock.t.len();
            let (Some(start), Some(last)) = (clock.t.first().copied(), clock.t.get(n - 1).copied()) else {
                break;
            };
            if last - start > self.max_step_seconds {
                self.skip(n, last);
                continue;
            }
            let mut step = Step {
                start,
                t: last,
                blocks: [(self.input_name(0), clock.block(0, n))].into(),
            };
            // How much longer than this step the spec lets a step be: another stream may start
            // that much after it and still belong to the step.
            let slack = self.max_step_seconds - (last - start);
            let (mut waiting, mut lacking) = (false, false);
            for (input, buffer) in self.inputs.iter().zip(&self.buffers).skip(1) {
                match input.take {
                    Take::Samples(k) => {
                        let from = buffer.t.partition_point(|t| *t < start);
                        if buffer.t.len() < from + k {
                            waiting = true;
                            lacking |= input.required;
                        } else if buffer.span(from, from + k) > self.max_step_seconds
                            || buffer.t.get(from).is_some_and(|first| first - start > slack)
                        {
                            // A hole in the stream, or its samples only begin after the step did.
                            lacking |= input.required;
                        } else {
                            step.blocks.insert(input.stream.clone(), buffer.block(from, from + k));
                        }
                    }
                    Take::Arrived => {
                        // Everything since the previous step, so nothing falls between two steps.
                        let from = match self.previous_end {
                            Some(end) => buffer.t.partition_point(|t| *t <= end),
                            None => buffer.t.partition_point(|t| *t < start),
                        };
                        let to = buffer.t.partition_point(|t| *t <= last);
                        step.blocks
                            .insert(input.stream.clone(), buffer.block(from, to.max(from)));
                    }
                }
            }
            if waiting && !end && held < 2 * n {
                break; // the other stream may simply not have arrived yet
            }
            if lacking {
                self.skip(n, last);
                continue;
            }
            if let Some(values) = self.model.step(&step).map_err(Error::Model)? {
                if values.len() != self.outputs {
                    return Err(Error::Spec(format!(
                        "the model returned {} values but names {}",
                        values.len(),
                        self.outputs
                    )));
                }
                out.push(Output { t: last, values });
            }
            self.drop_clock(n);
            self.previous_end = Some(last);
            self.trim(last - self.max_step_seconds);
        }
        Ok(out)
    }

    fn input_name(&self, index: usize) -> String {
        self.inputs.get(index).map(|i| i.stream.clone()).unwrap_or_default()
    }

    fn skip(&mut self, n: usize, last: f64) {
        self.drop_clock(n);
        self.restart();
        self.skipped += 1;
        self.trim(last - self.max_step_seconds);
    }
}
