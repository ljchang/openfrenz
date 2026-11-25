//! Signal-processing blocks and pipelines for Python.
//!
//! Arrays cross as raw bytes of 64-bit floats, one channel after another, and are wrapped by
//! NumPy on the Python side (`openfrenz.dsp`, `openfrenz.Pipeline`), as live chunks are.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};

use frenz_model::dsp::{self, Design, Start};
use frenz_model::{Block, Live, Model, Step, Take};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict};

use crate::{Band, FrenzError, wait};

fn floats(bytes: &[u8]) -> PyResult<Vec<f64>> {
    let (whole, rest) = bytes.as_chunks::<8>();
    if !rest.is_empty() {
        return Err(PyValueError::new_err("expected 64-bit floats"));
    }
    Ok(whole.iter().map(|b| f64::from_le_bytes(*b)).collect())
}

/// `channels` rows of equal length.
fn rows(bytes: &[u8], channels: usize) -> PyResult<Vec<Vec<f64>>> {
    let values = floats(bytes)?;
    if channels == 0 || values.len() % channels != 0 {
        return Err(PyValueError::new_err(
            "the data does not divide into that many channels",
        ));
    }
    let samples = values.len() / channels;
    Ok(if samples == 0 {
        vec![Vec::new(); channels]
    } else {
        values.chunks(samples).map(<[f64]>::to_vec).collect()
    })
}

fn bytes<'py>(py: Python<'py>, values: impl IntoIterator<Item = f64>) -> Bound<'py, PyByteArray> {
    let raw: Vec<u8> = values.into_iter().flat_map(f64::to_le_bytes).collect();
    PyByteArray::new(py, &raw)
}

fn flat<'py>(py: Python<'py>, rows: &[Vec<f64>]) -> Bound<'py, PyByteArray> {
    bytes(py, rows.iter().flatten().copied())
}

fn invalid(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A chain of filters that keeps its state between calls.
#[pyclass(module = "openfrenz._native", frozen)]
struct Filter {
    design: Design,
    filter: Mutex<dsp::Filter>,
}

#[pymethods]
impl Filter {
    #[new]
    fn new(steps: &str, rate_hz: f64, settled: bool) -> PyResult<Self> {
        let steps: Vec<serde_json::Value> = serde_json::from_str(steps).map_err(invalid)?;
        let design = Design::from_config(&steps, rate_hz).map_err(invalid)?;
        let start = if settled { Start::Settled } else { Start::Zero };
        Ok(Self {
            filter: Mutex::new(dsp::Filter::new(design.clone(), start)),
            design,
        })
    }

    fn process<'py>(&self, py: Python<'py>, data: &[u8], channels: usize) -> PyResult<Bound<'py, PyByteArray>> {
        let x = rows(data, channels)?;
        Ok(flat(py, &lock(&self.filter).process(&x)))
    }

    fn filtfilt<'py>(&self, py: Python<'py>, data: &[u8], channels: usize) -> PyResult<Bound<'py, PyByteArray>> {
        let x = rows(data, channels)?;
        let y: Vec<Vec<f64>> = x.iter().map(|c| dsp::filtfilt(&self.design, c)).collect();
        Ok(flat(py, &y))
    }

    fn reset(&self) {
        lock(&self.filter).reset();
    }
}

/// `(resolution_hz, density)` of one channel; averaged over stretches of `segment` if given.
#[pyfunction]
#[pyo3(signature = (data, rate_hz, segment=None))]
fn spectrum<'py>(
    py: Python<'py>,
    data: &[u8],
    rate_hz: f64,
    segment: Option<usize>,
) -> PyResult<(f64, Bound<'py, PyByteArray>)> {
    let x = floats(data)?;
    let spectrum = match segment {
        Some(segment) => dsp::welch(&x, rate_hz, segment),
        None => dsp::periodogram(&x, rate_hz),
    };
    Ok((spectrum.resolution_hz, bytes(py, spectrum.density)))
}

type Pair<'py> = (Bound<'py, PyByteArray>, Bound<'py, PyByteArray>);
type Scored<'py> = (Bound<'py, PyByteArray>, Bound<'py, PyByteArray>, BTreeMap<String, u64>);

/// `(amplitude, phase)` of one channel.
#[pyfunction]
fn analytic<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Pair<'py>> {
    let found = dsp::analytic(&floats(data)?);
    Ok((bytes(py, found.amplitude), bytes(py, found.phase)))
}

/// `(amplitude, phase)` of one channel at `hz`.
#[pyfunction]
fn morlet<'py>(py: Python<'py>, data: &[u8], rate_hz: f64, hz: f64, cycles: f64) -> PyResult<Pair<'py>> {
    let found = dsp::morlet(&floats(data)?, rate_hz, hz, cycles);
    Ok((bytes(py, found.amplitude), bytes(py, found.phase)))
}

/// `(resolution_hz, coherence per bin)` of two channels.
#[pyfunction]
fn coherence<'py>(
    py: Python<'py>,
    x: &[u8],
    y: &[u8],
    rate_hz: f64,
    segment: usize,
) -> PyResult<(f64, Bound<'py, PyByteArray>)> {
    let cross = dsp::cross_spectrum(&floats(x)?, &floats(y)?, rate_hz, segment);
    Ok((cross.resolution_hz, bytes(py, cross.coherence())))
}

/// `(modulation_index, mean_vector_length, preferred_phase)`.
#[pyfunction]
fn coupling(phase: &[u8], amplitude: &[u8], bins: usize) -> PyResult<(f64, f64, f64)> {
    let found = dsp::coupling(&floats(phase)?, &floats(amplitude)?, bins);
    Ok((found.modulation_index, found.mean_vector_length, found.preferred_phase))
}

/// Beats and heart-rate measures of one pulse channel.
#[pyfunction]
fn heart<'py>(py: Python<'py>, data: &[u8], rate_hz: f64) -> PyResult<Bound<'py, PyDict>> {
    let found = dsp::heart(&floats(data)?, rate_hz);
    let dict = PyDict::new(py);
    dict.set_item("beats", bytes(py, found.beats))?;
    dict.set_item("rate_bpm", found.rate_bpm)?;
    dict.set_item("sdnn_ms", found.sdnn_ms)?;
    dict.set_item("rmssd_ms", found.rmssd_ms)?;
    dict.set_item("quality", found.quality)?;
    Ok(dict)
}

/// `(mixing, threshold)` learned from clean data.
#[pyfunction]
fn asr_calibrate<'py>(py: Python<'py>, data: &[u8], channels: usize, rate_hz: f64, cutoff: f64) -> PyResult<Pair<'py>> {
    let reference = rows(data, channels)?;
    let calibration = py
        .detach(|| dsp::asr_calibrate(&reference, rate_hz, cutoff))
        .map_err(invalid)?;
    Ok((flat(py, &calibration.mixing), flat(py, &calibration.threshold)))
}

/// Artifact subspace reconstruction that keeps its state between calls.
#[pyclass(module = "openfrenz._native", frozen)]
struct Asr {
    asr: Mutex<dsp::Asr>,
}

#[pymethods]
impl Asr {
    #[new]
    fn new(mixing: &[u8], threshold: &[u8], channels: usize, max_removed: f64, window_s: f64) -> PyResult<Self> {
        let calibration = dsp::AsrCalibration {
            mixing: rows(mixing, channels)?,
            threshold: rows(threshold, channels)?,
        };
        Ok(Self {
            asr: Mutex::new(dsp::Asr::new(&calibration, max_removed, window_s).map_err(invalid)?),
        })
    }

    fn process<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        channels: usize,
        rate_hz: f64,
    ) -> PyResult<Bound<'py, PyByteArray>> {
        let x = rows(data, channels)?;
        Ok(flat(py, &lock(&self.asr).process(&x, rate_hz).map_err(invalid)?))
    }

    fn reset(&self) {
        lock(&self.asr).reset();
    }
}

pub(crate) fn model_error(error: frenz_model::Error) -> PyErr {
    match error {
        frenz_model::Error::Spec(_)
        | frenz_model::Error::Files(_)
        | frenz_model::Error::UnknownKind { .. }
        | frenz_model::Error::Access(_) => PyValueError::new_err(error.to_string()),
        other => FrenzError::new_err(other.to_string()),
    }
}

/// A model described by its files.
#[pyclass(module = "openfrenz._native", frozen)]
struct Pipeline {
    dir: PathBuf,
    model: Mutex<frenz_model::Pipeline>,
}

impl Pipeline {
    fn load(&self) -> PyResult<frenz_model::Pipeline> {
        frenz_model::Pipeline::from_dir(&self.dir).map_err(model_error)
    }
}

#[pymethods]
impl Pipeline {
    /// `source` is a folder, the name of a model that comes with openfrenz, or a Hugging Face
    /// repository (`owner/name` or `owner/name@revision`), read with `token` if given.
    #[new]
    #[pyo3(signature = (source, token = None))]
    fn new(py: Python<'_>, source: &str, token: Option<String>) -> PyResult<Self> {
        let dir = py
            .detach(|| frenz_model::resolve_with(source, token.as_deref()))
            .map_err(model_error)?;
        let model = frenz_model::Pipeline::from_dir(&dir).map_err(model_error)?;
        Ok(Self {
            dir,
            model: Mutex::new(model),
        })
    }

    #[getter]
    fn folder(&self) -> PathBuf {
        self.dir.clone()
    }

    /// A short name for the contents of the model's files; it changes whenever they do.
    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        frenz_model::fingerprint_dir(&self.dir).map_err(model_error)
    }

    #[getter]
    fn name(&self) -> String {
        lock(&self.model).spec().name.clone()
    }

    #[getter]
    fn version(&self) -> String {
        lock(&self.model).spec().version.clone()
    }

    /// `(stream, samples per step)` for every stream read; the first sets the steps.
    #[getter]
    fn inputs(&self) -> Vec<(String, usize)> {
        let model = lock(&self.model);
        let samples = |take| if let Take::Samples(n) = take { n } else { 0 };
        model
            .spec()
            .inputs
            .iter()
            .map(|i| (i.stream.clone(), samples(i.take)))
            .collect()
    }

    #[getter]
    fn outputs(&self) -> Vec<String> {
        lock(&self.model).spec().outputs.clone()
    }

    fn reset(&self) {
        lock(&self.model).reset();
    }

    /// One step from `(stream, data, channels)` blocks; `None` while blocks are warming up.
    fn step(&self, blocks: Vec<(String, Vec<u8>, usize)>) -> PyResult<Option<Vec<f64>>> {
        let mut parts = Vec::new();
        for (stream, data, channels) in blocks {
            let values = rows(&data, channels)?;
            let samples = values.first().map_or(0, Vec::len);
            parts.push((
                stream,
                Block {
                    t: vec![0.0; samples],
                    values,
                },
            ));
        }
        lock(&self.model)
            .step(&Step::new(0.0, 0.0, parts))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Runs a fresh copy of the model over a recording: `(times, values, steps skipped by the
    /// stream that was missing)`.
    fn run<'py>(&self, py: Python<'py>, recording: PathBuf) -> PyResult<Scored<'py>> {
        let model = self.load()?;
        let run = py
            .detach(|| frenz_model::run_recording(model, &recording))
            .map_err(model_error)?;
        let values = run.outputs.iter().flat_map(|o| o.values.iter().copied());
        Ok((
            bytes(py, run.outputs.iter().map(|o| o.t)),
            bytes(py, values),
            run.missed,
        ))
    }

    /// Runs a fresh copy of the model on `band` as it streams.
    fn stream(&self, band: Py<Band>) -> PyResult<ModelStream> {
        let (inner, reconnects) = band.get().model_feed()?;
        // The live runner spawns its reader on the runtime.
        let _guard = crate::runtime()?.enter();
        let live = Live::new(self.load()?, &inner, reconnects).map_err(model_error)?;
        Ok(ModelStream {
            live: Mutex::new(Some(live)),
            missed: Mutex::new(BTreeMap::new()),
            _band: band,
        })
    }
}

/// The time of a step and its values.
type Timed = (f64, Vec<f64>);

/// Iterator returned by `Pipeline.stream()`: `(time, values)` per step.
#[pyclass(module = "openfrenz._native", frozen)]
struct ModelStream {
    /// Taken out while a step is awaited, so the lock is not held across the wait.
    live: Mutex<Option<Live<frenz_model::Pipeline>>>,
    /// Steps skipped so far, by the stream that was missing.
    missed: Mutex<BTreeMap<String, u64>>,
    _band: Py<Band>,
}

#[pymethods]
impl ModelStream {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Steps skipped so far, counted by the stream that was missing from each.
    #[getter]
    fn missed(&self) -> BTreeMap<String, u64> {
        lock(&self.missed).clone()
    }

    /// The next step, waiting at most `timeout` seconds if one is given. `(False, None)` when
    /// no step completed in time, `(True, None)` when the band has ended.
    #[pyo3(signature = (timeout=None))]
    fn poll(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<(bool, Option<Timed>)> {
        let limit = timeout.map(|t| crate::seconds(t, "timeout")).transpose()?;
        let Some(mut live) = lock(&self.live).take() else {
            return Ok((true, None));
        };
        let result = wait(py, async {
            let output = match limit {
                Some(limit) => tokio::time::timeout(limit, live.next()).await.ok(),
                None => Some(live.next().await),
            };
            (live, output)
        });
        let (live, output) = result?;
        *lock(&self.missed) = live.missed().clone();
        match output {
            None => {
                *lock(&self.live) = Some(live);
                Ok((false, None))
            }
            Some(output) => match output.map_err(model_error)? {
                Some(output) => {
                    *lock(&self.live) = Some(live);
                    Ok((false, Some((output.t, output.values))))
                }
                None => Ok((true, None)),
            },
        }
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Timed>> {
        Ok(self.poll(py, None)?.1)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Filter>()?;
    module.add_class::<Asr>()?;
    module.add_class::<Pipeline>()?;
    module.add_class::<ModelStream>()?;
    for function in [
        wrap_pyfunction!(spectrum, module)?,
        wrap_pyfunction!(analytic, module)?,
        wrap_pyfunction!(morlet, module)?,
        wrap_pyfunction!(coherence, module)?,
        wrap_pyfunction!(coupling, module)?,
        wrap_pyfunction!(heart, module)?,
        wrap_pyfunction!(asr_calibrate, module)?,
    ] {
        module.add_function(function)?;
    }
    Ok(())
}
