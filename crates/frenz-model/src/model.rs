//! What a model is: the streams it reads, the steps it takes and the values it returns.

use std::collections::BTreeMap;

/// An error from inside a model.
pub type ModelError = Box<dyn std::error::Error + Send + Sync>;

/// How much of a stream one step holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Take {
    /// This many consecutive samples from the start of the step on. For streams with a steady
    /// rate. The samples must be free of holes and begin with the step, within the margin
    /// [`Spec::max_step_seconds`] leaves.
    Samples(usize),
    /// Whatever arrived during the step, which may be nothing. For streams without a steady rate
    /// (heart rate, SpO2, battery, contact) and for any stream a model wants untrimmed.
    Arrived,
}

/// One stream a model reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Input {
    /// Stream name, as in [`crate::streams::names`].
    pub stream: String,
    /// How much of it one step holds.
    pub take: Take,
    /// `true`: a step without this stream is skipped and the model restarts. `false`: the step
    /// runs without it. Only matters for [`Take::Samples`]; a stream taken as it arrives can
    /// always be empty.
    pub required: bool,
}

impl Input {
    /// A required stream, `count` samples per step.
    pub fn samples(stream: &str, count: usize) -> Self {
        Self {
            stream: stream.to_owned(),
            take: Take::Samples(count),
            required: true,
        }
    }

    /// A stream delivered as it arrives.
    pub fn arrived(stream: &str) -> Self {
        Self {
            stream: stream.to_owned(),
            take: Take::Arrived,
            required: false,
        }
    }

    /// The same input, used when present and done without otherwise.
    #[must_use]
    pub fn preferred(mut self) -> Self {
        self.required = false;
        self
    }
}

/// A model's description of itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    /// Model name.
    pub name: String,
    /// Model version.
    pub version: String,
    /// The streams it reads. The first sets the steps: it must be required and take a fixed
    /// number of samples, and every other stream is cut to match each of its steps.
    pub inputs: Vec<Input>,
    /// Names of the values a step returns, in order.
    pub outputs: Vec<String>,
    /// A step whose samples span more than this many seconds has a hole in it and is skipped.
    /// A little more than the step's nominal length.
    pub max_step_seconds: f64,
}

/// One stream's part of a step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Block {
    /// Estimated acquisition time of each sample (Unix seconds).
    pub t: Vec<f64>,
    /// `values[channel][sample]`, raw counts, channels in the stream's order
    /// ([`crate::streams::channels`]).
    pub values: Vec<Vec<f64>>,
}

/// The data of one step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Step {
    /// Time of the first sample of the step (Unix seconds).
    pub start: f64,
    /// Time of the last sample of the step (Unix seconds).
    pub t: f64,
    pub(crate) blocks: BTreeMap<String, Block>,
}

impl Step {
    /// A step from its parts, for testing a model directly.
    pub fn new(start: f64, t: f64, blocks: impl IntoIterator<Item = (String, Block)>) -> Self {
        Self {
            start,
            t,
            blocks: blocks.into_iter().collect(),
        }
    }

    /// The step's data for `stream`. `None` for a stream the model did not ask for and for a
    /// preferred stream that is absent from this step.
    pub fn get(&self, stream: &str) -> Option<&Block> {
        self.blocks.get(stream)
    }
}

/// The values a model returned for one step.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// Time of the last sample of the step (Unix seconds).
    pub t: f64,
    /// One value per name in [`Spec::outputs`].
    pub values: Vec<f64>,
}

/// A model that is run one step at a time and keeps its own state between steps.
pub trait Model: Send {
    /// What the model reads and returns.
    fn spec(&self) -> &Spec;

    /// Forgets everything carried from earlier steps. Called before the first step and whenever
    /// the next step does not directly follow the previous one.
    fn reset(&mut self);

    /// Takes one step. `Ok(None)` means no output yet (still warming up).
    fn step(&mut self, step: &Step) -> Result<Option<Vec<f64>>, ModelError>;
}

impl<M: Model + ?Sized> Model for Box<M> {
    fn spec(&self) -> &Spec {
        (**self).spec()
    }
    fn reset(&mut self) {
        (**self).reset();
    }
    fn step(&mut self, step: &Step) -> Result<Option<Vec<f64>>, ModelError> {
        (**self).step(step)
    }
}
