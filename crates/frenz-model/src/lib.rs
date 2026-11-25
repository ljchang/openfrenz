//! Run step-based models on FRENZ brainband data, from recordings or live.
//!
//! A [`Model`] says which streams it reads and how much of each makes one step ([`Spec`]), and
//! turns each step into named output values. The same model runs on a recording
//! ([`run_recording`]) and on a connected or replayed band ([`Live`]); [`Stepper`] cuts the
//! streams into steps for both and restarts the model wherever data was lost.
//!
//! Models can be written in code against [`Model`], using the blocks in [`dsp`], or entirely as
//! data: a [`Pipeline`] is a `config.json` that wires standard blocks together.
//!
//! Model files are a folder with `config.json` and usually `model.safetensors`. `config.json`
//! names the kind of model under the key `openfrenz_model`; a [`Registry`] maps each kind to the
//! code that loads it. With the `hub` feature the folder can be fetched from the Hugging Face Hub
//! by repository name.
//!
//! ```
//! use frenz_model::{Input, Model, ModelError, Spec, Step};
//!
//! /// Root mean square of each EEG channel over one second.
//! struct Rms(Spec);
//!
//! impl Rms {
//!     fn new() -> Self {
//!         Self(Spec {
//!             name: "rms".into(),
//!             version: "1".into(),
//!             inputs: vec![Input::samples("eeg", 125)],
//!             outputs: ["forehead_right", "comb_left", "forehead_left", "comb_right"].map(String::from).into(),
//!             max_step_seconds: 1.1,
//!         })
//!     }
//! }
//!
//! impl Model for Rms {
//!     fn spec(&self) -> &Spec {
//!         &self.0
//!     }
//!     fn reset(&mut self) {}
//!     fn step(&mut self, step: &Step) -> Result<Option<Vec<f64>>, ModelError> {
//!         let eeg = step.get("eeg").ok_or("no eeg")?;
//!         let rms = |x: &Vec<f64>| (x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64).sqrt();
//!         Ok(Some(eeg.values.iter().map(rms).collect()))
//!     }
//! }
//! # let _ = Rms::new();
//! ```

pub mod dsp;
mod export;
mod load;
mod model;
mod pipeline;
mod run;
mod stepper;
pub mod streams;

pub use export::{Exported, export_models};
#[cfg(feature = "hub")]
pub use load::fetch;
pub use load::{BUILT_IN, CONFIG_KEY, Registry, fingerprint, fingerprint_dir, kind, resolve};
pub use model::{Block, Input, Model, ModelError, Output, Spec, Step, Take};
pub use pipeline::Pipeline;
pub use run::{Live, Run, run_recording, run_session};
pub use stepper::Stepper;

/// What can go wrong running a model.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The model's description of itself cannot be used.
    #[error("model spec: {0}")]
    Spec(String),
    /// The data has no stream the model requires.
    #[error("the data has no {0} stream, which the model requires")]
    MissingStream(String),
    /// The model failed on a step.
    #[error("model: {0}")]
    Model(#[source] ModelError),
    /// No code is registered for the kind of model the files name.
    #[error("needs the model kind {kind:?}; this program has: {known}")]
    UnknownKind {
        /// The kind `config.json` names.
        kind: String,
        /// The kinds this program can load, comma-separated.
        known: String,
    },
    /// The model files are missing or malformed.
    #[error("model files: {0}")]
    Files(String),
    /// The model files could not be fetched.
    #[error("fetching model files: {0}")]
    Fetch(String),
    /// Reading a recording failed.
    #[error(transparent)]
    Export(#[from] frenz_export::Error),
    /// The band failed.
    #[error(transparent)]
    Band(#[from] frenz::Error),
    /// File access failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSON failed to parse.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
