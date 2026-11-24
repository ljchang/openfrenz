//! Signal-processing building blocks for models: filters, spectra, wavelets, envelopes,
//! summaries, coherence between channels, coupling between rhythms, and heartbeats from a pulse
//! signal.
//!
//! Each block comes in the form a step-by-step model needs. Filters carry their state from one
//! call to the next, so filtering a signal in pieces gives the same numbers as filtering it
//! whole. The numerics follow SciPy's (`butter`, `sosfilt`, `sosfiltfilt`, `iirnotch`,
//! `periodogram`, `welch`, `hilbert`, `coherence`) and are tested against values computed with it.

mod asr;
mod envelope;
mod filter;
mod heart;
mod relate;
mod spectrum;
mod stats;
mod wavelet;

pub use asr::{Asr, AsrCalibration, asr_calibrate};
pub use envelope::{Analytic, analytic};
pub use filter::{Design, Filter, Section, Start, filtfilt};
pub use heart::{Heart, heart};
pub use relate::{Coupling, CouplingSums, CrossSpectrum, Recent, correlation, coupling, cross_spectrum};
pub use spectrum::{Spectrum, periodogram, welch};
pub use stats::{Quality, Summary, quality, summary};
pub use wavelet::{morlet, scalogram};
