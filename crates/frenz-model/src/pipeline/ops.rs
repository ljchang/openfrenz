//! The blocks a pipeline is made of.

// Channels and pairs are looked up by indexes found in the same signal just before.
#![allow(clippy::indexing_slicing)]

use serde_json::Value as Json;

use super::nn::{Network, Weights};
use crate::dsp::{
    Asr, AsrCalibration, CouplingSums, CrossSpectrum, Design, Filter, JumpSettings, Jumps, PeakSettings, Peaks, Recent,
    Start, analytic, cross_spectrum, filtfilt, heart, morlet, periodogram, quality, summary, welch,
};

/// Samples of one stream or of something made from one.
#[derive(Debug, Clone)]
pub(crate) struct Signal {
    pub(crate) rate_hz: f64,
    pub(crate) channels: Vec<String>,
    /// `data[channel][sample]`.
    pub(crate) data: Vec<Vec<f64>>,
}

/// What flows from one node to the next.
#[derive(Debug, Clone)]
pub(crate) enum Value {
    Signal(Signal),
    /// Named numbers describing one step.
    Features {
        names: Vec<String>,
        values: Vec<f64>,
    },
    /// The features of several steps, oldest first.
    Sequence {
        rows: Vec<Vec<f64>>,
    },
}

/// One node's work. `Ok(None)` means it has not seen enough steps yet.
pub(crate) trait Op: Send {
    fn reset(&mut self);
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String>;
}

fn one_signal<'a>(inputs: &[&'a Value]) -> Result<&'a Signal, String> {
    match inputs {
        [Value::Signal(signal)] => Ok(signal),
        _ => Err("takes one signal".into()),
    }
}

fn number(config: &Json, key: &str) -> Option<f64> {
    config.get(key).and_then(Json::as_f64)
}

/// The largest count a config may ask for: steps to keep, samples in a stretch, phase bins.
/// Configs can come from anywhere, so nothing they say may make a block arbitrarily large.
const MAX_COUNT: usize = 100_000;

fn count(config: &Json, key: &str, default: usize) -> Result<usize, String> {
    match config.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_u64()
            .map(|n| n as usize)
            .filter(|n| (1..=MAX_COUNT).contains(n))
            .ok_or_else(|| format!("{key} must be a whole number between 1 and {MAX_COUNT}")),
    }
}

/// A number of seconds that is neither negative nor absurd; 0 if not given.
fn duration(config: &Json, key: &str) -> Result<f64, String> {
    match config.get(key) {
        None => Ok(0.0),
        Some(v) => v
            .as_f64()
            .filter(|s| (0.0..=3600.0).contains(s))
            .ok_or_else(|| format!("{key} must be a number of seconds between 0 and 3600")),
    }
}

fn flag(config: &Json, key: &str) -> bool {
    config.get(key).and_then(Json::as_bool).unwrap_or(false)
}

fn band(value: &Json) -> Result<(f64, f64), String> {
    match value.as_array().map(Vec::as_slice) {
        Some([low, high]) => match (low.as_f64(), high.as_f64()) {
            (Some(low), Some(high)) if low < high => Ok((low, high)),
            _ => Err(format!("{value} is not a band [low, high]")),
        },
        _ => Err(format!("{value} is not a band [low, high]")),
    }
}

fn bands(config: &Json, key: &str) -> Result<Vec<(f64, f64)>, String> {
    config
        .get(key)
        .and_then(Json::as_array)
        .ok_or_else(|| format!("needs {key}: a list of [low, high]"))?
        .iter()
        .map(band)
        .collect()
}

fn strings(config: &Json, key: &str, default: &[&str]) -> Result<Vec<String>, String> {
    match config.get(key) {
        None => Ok(default.iter().map(|s| (*s).to_owned()).collect()),
        Some(list) => list
            .as_array()
            .ok_or_else(|| format!("{key} must be a list of names"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("{key} must be a list of names"))
            })
            .collect(),
    }
}

fn label((low, high): (f64, f64)) -> String {
    format!("{low}-{high}")
}

fn channel_index(signal: &Signal, name: &str) -> Result<usize, String> {
    signal
        .channels
        .iter()
        .position(|c| c == name)
        .ok_or_else(|| format!("no channel {name:?} among {:?}", signal.channels))
}

/// `select`: some channels of a signal, in the order given.
struct Select {
    channels: Vec<String>,
}

impl Op for Select {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let mut data = Vec::new();
        for name in &self.channels {
            data.push(signal.data[channel_index(signal, name)?].clone());
        }
        Ok(Some(Value::Signal(Signal {
            rate_hz: signal.rate_hz,
            channels: self.channels.clone(),
            data,
        })))
    }
}

/// `filter`: a chain of filters, carried across steps or applied both ways within each step.
struct FilterOp {
    steps: Vec<Json>,
    start: Start,
    zero_phase: bool,
    built: Option<(f64, Design, Filter)>,
}

impl Op for FilterOp {
    fn reset(&mut self) {
        if let Some((_, _, filter)) = &mut self.built {
            filter.reset();
        }
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        if self.built.as_ref().is_none_or(|(rate, _, _)| *rate != signal.rate_hz) {
            let chain = Design::from_config(&self.steps, signal.rate_hz).map_err(|e| e.to_string())?;
            self.built = Some((signal.rate_hz, chain.clone(), Filter::new(chain, self.start)));
        }
        let Some((_, chain, filter)) = &mut self.built else {
            return Ok(None);
        };
        let data = if self.zero_phase {
            signal.data.iter().map(|x| filtfilt(chain, x)).collect()
        } else {
            filter.process(&signal.data)
        };
        Ok(Some(Value::Signal(Signal {
            rate_hz: signal.rate_hz,
            channels: signal.channels.clone(),
            data,
        })))
    }
}

/// `band_power`: power (or magnitude) in each band of each channel.
struct BandPower {
    bands: Vec<(f64, f64)>,
    relative_to: Option<(f64, f64)>,
    log: bool,
    segment: Option<usize>,
    /// The size of each frequency as magnitude instead of power.
    magnitude: bool,
    /// Integrate bands with Simpson's rule over the bins nearest their edges, instead of summing bins.
    simpson: bool,
    hann: bool,
    remove_mean: bool,
}

impl Op for BandPower {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let what = if self.magnitude { "magnitude" } else { "power" };
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for (channel, x) in signal.channels.iter().zip(&signal.data) {
            let spectrum = match (self.magnitude, self.segment) {
                (true, _) => crate::dsp::magnitude(x, signal.rate_hz, self.hann, self.remove_mean),
                (false, Some(segment)) => welch(x, signal.rate_hz, segment),
                (false, None) => periodogram(x, signal.rate_hz),
            };
            let integrate = |(low, high): (f64, f64)| {
                if self.simpson {
                    spectrum.band_simpson(low, high)
                } else {
                    spectrum.band_power(low, high)
                }
            };
            let total = self.relative_to.map(integrate);
            for b in &self.bands {
                let power = integrate(*b) / total.unwrap_or(1.0);
                names.push(format!("{channel}:{what}:{}", label(*b)));
                values.push(if self.log { power.ln() } else { power });
            }
        }
        Ok(Some(Value::Features { names, values }))
    }
}

/// `envelope`: level and spread of the amplitude envelope of one band of each channel.
struct Envelope {
    band: (f64, f64),
    order: usize,
    stats: Vec<String>,
}

fn stat(x: &[f64], name: &str) -> Result<f64, String> {
    if x.is_empty() {
        // Nothing arrived during the step.
        return Ok(f64::NAN);
    }
    let s = summary(x);
    Ok(match name {
        "mean" => s.mean,
        "sd" => s.sd,
        // Sizes that span orders of magnitude, such as muscle activity, are steadier as logarithms.
        "log_sd" => s.sd.max(f64::MIN_POSITIVE).ln(),
        "range" => s.range,
        "largest_step" => s.largest_step,
        other => return Err(format!("unknown statistic {other:?}")),
    })
}

impl Op for Envelope {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let design =
            Design::bandpass(self.order, self.band.0, self.band.1, signal.rate_hz).map_err(|e| e.to_string())?;
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for (channel, x) in signal.channels.iter().zip(&signal.data) {
            let amplitude = analytic(&filtfilt(&design, x)).amplitude;
            for name in &self.stats {
                names.push(format!("{channel}:envelope:{}:{name}", label(self.band)));
                values.push(stat(&amplitude, name)?);
            }
        }
        Ok(Some(Value::Features { names, values }))
    }
}

/// `summary`: level and spread of each channel.
struct Summary {
    stats: Vec<String>,
}

impl Op for Summary {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for (channel, x) in signal.channels.iter().zip(&signal.data) {
            for name in &self.stats {
                names.push(format!("{channel}:{name}"));
                values.push(stat(x, name)?);
            }
        }
        Ok(Some(Value::Features { names, values }))
    }
}

/// `quality`: the fractions of repeated, clipped and invalid samples of each channel.
struct Quality {
    limit: f64,
}

impl Op for Quality {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for (channel, x) in signal.channels.iter().zip(&signal.data) {
            let q = quality(x, self.limit);
            for (name, value) in [("repeated", q.repeated), ("clipped", q.clipped), ("invalid", q.invalid)] {
                names.push(format!("{channel}:{name}"));
                values.push(value);
            }
        }
        Ok(Some(Value::Features { names, values }))
    }
}

/// `coherence`: coherence of pairs of channels in bands, over the last few steps.
struct Coherence {
    pairs: Vec<(String, String)>,
    bands: Vec<(f64, f64)>,
    segment: usize,
    recent: Vec<Recent<CrossSpectrum>>,
}

impl Op for Coherence {
    fn reset(&mut self) {
        self.recent.iter_mut().for_each(Recent::clear);
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let (mut names, mut values) = (Vec::new(), Vec::new());
        let mut ready = true;
        for ((a, b), recent) in self.pairs.iter().zip(&mut self.recent) {
            let (x, y) = (
                &signal.data[channel_index(signal, a)?],
                &signal.data[channel_index(signal, b)?],
            );
            recent.push(cross_spectrum(x, y, signal.rate_hz, self.segment));
            ready &= recent.is_full();
            let Some(total) = recent.total() else { continue };
            for band in &self.bands {
                names.push(format!("{a}~{b}:coherence:{}", label(*band)));
                values.push(total.band_coherence(band.0, band.1));
            }
        }
        Ok(ready.then_some(Value::Features { names, values }))
    }
}

/// `coupling`: how the size of a fast rhythm follows the phase of a slow one in each channel,
/// over the last few steps.
struct Coupling {
    phase_band: (f64, f64),
    amplitude_band: (f64, f64),
    order: usize,
    bins: usize,
    trim_s: f64,
    steps: usize,
    recent: Vec<Recent<CouplingSums>>,
}

impl Op for Coupling {
    fn reset(&mut self) {
        self.recent.clear();
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let slow = Design::bandpass(self.order, self.phase_band.0, self.phase_band.1, signal.rate_hz)
            .map_err(|e| e.to_string())?;
        let fast = Design::bandpass(self.order, self.amplitude_band.0, self.amplitude_band.1, signal.rate_hz)
            .map_err(|e| e.to_string())?;
        if self.recent.len() != signal.data.len() {
            self.recent = signal.data.iter().map(|_| Recent::new(self.steps)).collect();
        }
        let (mut names, mut values) = (Vec::new(), Vec::new());
        let mut ready = true;
        let what = format!("{}/{}", label(self.phase_band), label(self.amplitude_band));
        for ((channel, x), recent) in signal.channels.iter().zip(&signal.data).zip(&mut self.recent) {
            let phase = analytic(&filtfilt(&slow, x)).phase;
            let amplitude = analytic(&filtfilt(&fast, x)).amplitude;
            // The ends of each step are distorted by the filters and are left out.
            let trim = ((self.trim_s * signal.rate_hz).round() as usize).min(x.len() / 2);
            let keep = trim..x.len() - trim;
            recent.push(CouplingSums::new(&phase[keep.clone()], &amplitude[keep], self.bins));
            ready &= recent.is_full();
            let Some(total) = recent.total() else { continue };
            let measured = total.measure();
            names.push(format!("{channel}:coupling:{what}:modulation_index"));
            values.push(measured.modulation_index);
            names.push(format!("{channel}:coupling:{what}:mean_vector_length"));
            values.push(measured.mean_vector_length);
        }
        Ok(ready.then_some(Value::Features { names, values }))
    }
}

/// `wavelet`: the mean size of each channel at each frequency over the step.
struct Wavelet {
    frequencies: Vec<f64>,
    cycles: f64,
    trim_s: f64,
    log: bool,
}

/// Most multiply-adds a wavelet node may spend on one step. A wavelet is as long as its cycles
/// over its frequency, so without a limit a config could ask for hours of work per step.
const WAVELET_BUDGET: f64 = 2e8;

impl Op for Wavelet {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let samples = signal.data.first().map_or(0, Vec::len) as f64;
        // Samples in each wavelet (eight spreads of its Gaussian), summed over the frequencies.
        let lengths: f64 = self
            .frequencies
            .iter()
            .map(|hz| 8.0 * self.cycles / (2.0 * std::f64::consts::PI * hz) * signal.rate_hz)
            .sum();
        if samples * lengths * signal.data.len() as f64 > WAVELET_BUDGET {
            return Err(
                "these frequencies and cycles are too much work for one step: use fewer, or higher, frequencies".into(),
            );
        }
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for (channel, x) in signal.channels.iter().zip(&signal.data) {
            let trim = ((self.trim_s * signal.rate_hz).round() as usize).min(x.len() / 2);
            for hz in &self.frequencies {
                let amplitude = morlet(x, signal.rate_hz, *hz, self.cycles).amplitude;
                let kept = &amplitude[trim..amplitude.len() - trim];
                let mean = kept.iter().sum::<f64>() / kept.len().max(1) as f64;
                names.push(format!("{channel}:wavelet:{hz}"));
                values.push(if self.log { mean.ln() } else { mean });
            }
        }
        Ok(Some(Value::Features { names, values }))
    }
}

/// `heart`: heart rate and its variability from one pulse channel, over the last few steps.
struct HeartOp {
    channel: Option<String>,
    invert: bool,
    recent: Recent<Vec<f64>>,
}

impl Op for HeartOp {
    fn reset(&mut self) {
        self.recent.clear();
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let index = match &self.channel {
            Some(name) => channel_index(signal, name)?,
            None => 0,
        };
        let x = signal.data.get(index).ok_or("the signal has no channels")?;
        self.recent.push(if self.invert {
            x.iter().map(|v| -v).collect()
        } else {
            x.clone()
        });
        if !self.recent.is_full() {
            return Ok(None);
        }
        let joined: Vec<f64> = self.recent.iter().flatten().copied().collect();
        let found = heart(&joined, signal.rate_hz);
        Ok(Some(Value::Features {
            names: ["rate_bpm", "sdnn_ms", "rmssd_ms", "quality"]
                .iter()
                .map(|n| format!("heart:{n}"))
                .collect(),
            values: vec![found.rate_bpm, found.sdnn_ms, found.rmssd_ms, found.quality],
        }))
    }
}

/// `asr`: artifact subspace reconstruction with a stored calibration, optionally after a filter chain of its own
/// (`before`), the whole run `passes` times over each step with the first result kept.
struct AsrOp {
    asr: Asr,
    before: Option<FilterOp>,
    passes: usize,
}

impl Op for AsrOp {
    fn reset(&mut self) {
        self.asr.reset();
        if let Some(before) = &mut self.before {
            before.reset();
        }
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let mut first = None;
        for _ in 0..self.passes {
            let filtered = match &mut self.before {
                Some(before) => match before.run(inputs)? {
                    Some(Value::Signal(filtered)) => filtered.data,
                    _ => return Err("the filter before the reconstruction gave nothing".into()),
                },
                None => signal.data.clone(),
            };
            let cleaned = self.asr.process(&filtered, signal.rate_hz).map_err(|e| e.to_string())?;
            first.get_or_insert(cleaned);
        }
        Ok(Some(Value::Signal(Signal {
            rate_hz: signal.rate_hz,
            channels: signal.channels.clone(),
            data: first.unwrap_or_default(),
        })))
    }
}

/// A filter chain's `start`: `zero`, `settled`, or a number (that value, held for ever).
fn start(config: &Json, key: &str) -> Result<Start, String> {
    match config.get(key) {
        None => Ok(Start::Zero),
        Some(Json::String(s)) if s == "zero" => Ok(Start::Zero),
        Some(Json::String(s)) if s == "settled" => Ok(Start::Settled),
        Some(v) => v
            .as_f64()
            .filter(|x| x.is_finite())
            .map(Start::Value)
            .ok_or_else(|| format!("{key} must be zero, settled or a number, not {v}")),
    }
}

fn numbers(value: &Json, what: &str) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{what} must be a list of numbers"))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| format!("{what} must be a list of numbers")))
        .collect()
}

/// `concat`: several sets of features, one after another.
struct Concat;

impl Op for Concat {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let (mut all_names, mut all_values) = (Vec::new(), Vec::new());
        for input in inputs {
            let Value::Features { names, values } = input else {
                return Err("takes features only".into());
            };
            all_names.extend(names.iter().cloned());
            all_values.extend(values);
        }
        Ok(Some(Value::Features {
            names: all_names,
            values: all_values,
        }))
    }
}

/// `normalize`: features minus a centre, over a scale; either stored or tracked as steps pass.
enum Normalize {
    Stored {
        center: Vec<f64>,
        scale: Vec<f64>,
    },
    Running {
        /// Weight of the newest step.
        rate: f64,
        warmup: usize,
        seen: usize,
        mean: Vec<f64>,
        variance: Vec<f64>,
    },
}

impl Op for Normalize {
    fn reset(&mut self) {
        if let Self::Running {
            seen, mean, variance, ..
        } = self
        {
            *seen = 0;
            mean.clear();
            variance.clear();
        }
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let [Value::Features { names, values }] = inputs else {
            return Err("takes one set of features".into());
        };
        let normalized: Vec<f64> = match self {
            Self::Stored { center, scale } => {
                if center.len() != values.len() || scale.len() != values.len() {
                    return Err(format!(
                        "was stored for {} features and given {}",
                        center.len(),
                        values.len()
                    ));
                }
                values
                    .iter()
                    .zip(center.iter().zip(scale.iter()))
                    .map(|(v, (c, s))| (v - c) / s)
                    .collect()
            }
            Self::Running {
                rate,
                warmup,
                seen,
                mean,
                variance,
            } => {
                if mean.len() != values.len() {
                    *mean = values.clone();
                    *variance = vec![0.0; values.len()];
                    *seen = 0;
                }
                // Early on the plain average of the steps so far, then a fading one.
                let weight = rate.max(1.0 / (*seen as f64 + 1.0));
                for ((m, var), v) in mean.iter_mut().zip(variance.iter_mut()).zip(values) {
                    let off = v - *m;
                    *m += weight * off;
                    *var = (1.0 - weight) * (*var + weight * off * off);
                }
                *seen += 1;
                if *seen <= *warmup {
                    return Ok(None);
                }
                values
                    .iter()
                    .zip(mean.iter().zip(variance.iter()))
                    .map(|(v, (m, var))| (v - m) / var.sqrt().max(1e-12))
                    .collect()
            }
        };
        Ok(Some(Value::Features {
            names: names.clone(),
            values: normalized,
        }))
    }
}

/// `history`: the features of the last few steps, once that many have passed.
struct History {
    recent: Recent<Vec<f64>>,
}

impl Op for History {
    fn reset(&mut self) {
        self.recent.clear();
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let [Value::Features { values, .. }] = inputs else {
            return Err("takes one set of features".into());
        };
        self.recent.push(values.clone());
        Ok(self.recent.is_full().then(|| Value::Sequence {
            rows: self.recent.iter().cloned().collect(),
        }))
    }
}

/// `network`: stored layers applied to features or to a history of them.
struct NetworkOp {
    network: Network,
    names: Vec<String>,
}

impl Op for NetworkOp {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let rows = match inputs {
            [Value::Features { values, .. }] => vec![values.clone()],
            [Value::Sequence { rows }] => rows.clone(),
            _ => return Err("takes features or a history of them".into()),
        };
        let mut out = self.network.forward(rows)?;
        let (Some(values), true) = (out.pop(), out.is_empty()) else {
            return Err("must end in a single row: add a last, mean or flatten layer".into());
        };
        if values.len() != self.names.len() {
            return Err(format!(
                "returns {} values but names {}",
                values.len(),
                self.names.len()
            ));
        }
        Ok(Some(Value::Features {
            names: self.names.clone(),
            values,
        }))
    }
}

/// A channel made by `mix`: its name and the weight it gives to each channel it reads.
type Mixed = (String, Vec<(String, f64)>);

/// `mix`: new channels, each a weighted sum of the signal's channels.
struct Mix {
    channels: Vec<Mixed>,
}

impl Op for Mix {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let samples = signal.data.first().map_or(0, Vec::len);
        let mut data = Vec::with_capacity(self.channels.len());
        for (_, weights) in &self.channels {
            let mut mixed = vec![0.0; samples];
            for (name, weight) in weights {
                for (m, x) in mixed.iter_mut().zip(&signal.data[channel_index(signal, name)?]) {
                    *m += weight * x;
                }
            }
            data.push(mixed);
        }
        Ok(Some(Value::Signal(Signal {
            rate_hz: signal.rate_hz,
            channels: self.channels.iter().map(|(name, _)| name.clone()).collect(),
            data,
        })))
    }
}

/// Settings of `peaks` and `jumps` given in seconds, turned into samples once the rate is known.
#[derive(Clone, Copy)]
struct Seconds {
    threshold: f64,
    back: f64,
    ahead: f64,
    min_gap: f64,
    near: f64,
    far: f64,
    leak: f64,
    scale_half_life: f64,
    warmup: f64,
}

/// At least one sample, however short the time.
fn in_samples(seconds: f64, rate_hz: f64) -> usize {
    (seconds * rate_hz).round().max(1.0) as usize
}

/// `peaks`: peaks of each channel that stand out from their surroundings.
struct PeaksOp {
    seconds: Seconds,
    built: Option<(f64, Vec<Peaks>)>,
}

impl Op for PeaksOp {
    fn reset(&mut self) {
        if let Some((_, detectors)) = &mut self.built {
            detectors.iter_mut().for_each(Peaks::reset);
        }
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let rate = signal.rate_hz;
        if self
            .built
            .as_ref()
            .is_none_or(|(built, detectors)| *built != rate || detectors.len() != signal.data.len())
        {
            let s = self.seconds;
            let settings = PeakSettings {
                threshold: s.threshold,
                back: in_samples(s.back, rate),
                ahead: in_samples(s.ahead, rate),
                min_gap: in_samples(s.min_gap, rate),
                scale_half_life: s.scale_half_life * rate,
                warmup: in_samples(s.warmup, rate),
            };
            self.built = Some((rate, signal.data.iter().map(|_| Peaks::new(settings)).collect()));
        }
        let Some((_, detectors)) = &mut self.built else {
            return Ok(None);
        };
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for ((channel, x), detector) in signal.channels.iter().zip(&signal.data).zip(detectors) {
            let (mut count, mut largest) = (0.0, None::<crate::dsp::Peak>);
            for v in x {
                if let Some(peak) = detector.push(*v) {
                    count += 1.0;
                    if largest.is_none_or(|l| peak.size > l.size) {
                        largest = Some(peak);
                    }
                }
            }
            // When the largest peak was, counted back from the step's last sample.
            let last = detector.seen().saturating_sub(1);
            let when = largest.map_or(f64::NAN, |p| -((last - p.index.min(last)) as f64) / rate);
            for (name, value) in [
                ("peaks", count),
                ("peak_time", when),
                ("peak_size", largest.map_or(f64::NAN, |p| p.size)),
            ] {
                names.push(format!("{channel}:{name}"));
                values.push(value);
            }
        }
        Ok(Some(Value::Features { names, values }))
    }
}

/// `jumps`: where each channel stands after its abrupt changes of level, fading back to zero.
struct JumpsOp {
    seconds: Seconds,
    built: Option<(f64, Vec<Jumps>)>,
}

impl Op for JumpsOp {
    fn reset(&mut self) {
        if let Some((_, followers)) = &mut self.built {
            followers.iter_mut().for_each(Jumps::reset);
        }
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let rate = signal.rate_hz;
        if self
            .built
            .as_ref()
            .is_none_or(|(built, followers)| *built != rate || followers.len() != signal.data.len())
        {
            let s = self.seconds;
            let settings = JumpSettings {
                threshold: s.threshold,
                min_gap: in_samples(s.min_gap, rate),
                near: in_samples(s.near, rate),
                far: in_samples(s.far, rate),
                leak_half_life: s.leak * rate,
                scale_half_life: s.scale_half_life * rate,
                warmup: in_samples(s.warmup, rate),
            };
            self.built = Some((rate, signal.data.iter().map(|_| Jumps::new(settings)).collect()));
        }
        let Some((_, followers)) = &mut self.built else {
            return Ok(None);
        };
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for ((channel, x), follower) in signal.channels.iter().zip(&signal.data).zip(followers) {
            let count = x.iter().filter(|v| follower.push(**v)).count();
            names.push(format!("{channel}:position"));
            values.push(follower.position());
            names.push(format!("{channel}:jumps"));
            values.push(count as f64);
        }
        Ok(Some(Value::Features { names, values }))
    }
}

/// `tilt`: how the head is held, from the direction of gravity in an accelerometer's `x`, `y`
/// and `z` (the band's axes: x down toward the chin, y toward the wearer's left, z forward).
struct Tilt {
    counts_per_g: f64,
}

impl Op for Tilt {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let mean = |name: &str| -> Result<f64, String> {
            let x = &signal.data[channel_index(signal, name)?];
            Ok(if x.is_empty() {
                f64::NAN
            } else {
                x.iter().sum::<f64>() / x.len() as f64
            })
        };
        // At rest the sensor reads the push that holds it up, so its reading points up.
        let (x, y, z) = (mean("x")?, mean("y")?, mean("z")?);
        let size = (x * x + y * y + z * z).sqrt();
        // How far each of the head's axes leans from level, and how far its top is from straight up.
        let lean = |part: f64| (part / size).clamp(-1.0, 1.0);
        let values = vec![
            lean(-z).asin().to_degrees(),
            lean(-y).asin().to_degrees(),
            lean(-x).acos().to_degrees(),
            size / self.counts_per_g,
        ];
        let names = ["pitch_deg", "roll_deg", "from_upright_deg", "g"];
        Ok(Some(Value::Features {
            names: names.iter().map(|n| format!("tilt:{n}")).collect(),
            values,
        }))
    }
}

/// `ratios`: one measure over another, for every channel that has both. A pair `[a, b]` matches each feature
/// named `<prefix>:a` with `<prefix>:b` (or `a` with `b` exactly) and gives `<prefix>:a/b`.
struct Ratios {
    pairs: Vec<(String, String)>,
    log: bool,
}

impl Op for Ratios {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let [Value::Features { names, values }] = inputs else {
            return Err("takes one set of features".into());
        };
        let (mut out_names, mut out_values) = (Vec::new(), Vec::new());
        for (a, b) in &self.pairs {
            let mut found = false;
            for (name, top) in names.iter().zip(values) {
                let prefix = if name == a {
                    ""
                } else if let Some(prefix) = name.strip_suffix(a.as_str()).and_then(|p| p.strip_suffix(':')) {
                    prefix
                } else {
                    continue;
                };
                let partner = if prefix.is_empty() {
                    b.clone()
                } else {
                    format!("{prefix}:{b}")
                };
                let Some(k) = names.iter().position(|n| *n == partner) else {
                    continue;
                };
                found = true;
                let ratio = top / values[k];
                out_names.push(if prefix.is_empty() {
                    format!("{a}/{b}")
                } else {
                    format!("{prefix}:{a}/{b}")
                });
                out_values.push(if self.log { ratio.ln() } else { ratio });
            }
            if !found {
                return Err(format!("no features named {a:?} with a matching {b:?}"));
            }
        }
        Ok(Some(Value::Features {
            names: out_names,
            values: out_values,
        }))
    }
}

/// `average`: each measure averaged over channels. Features named `<channel>:<measure>` from the listed
/// channels (all channels if none are listed) become `<name>:<measure>`.
struct Average {
    channels: Vec<String>,
    name: String,
}

impl Op for Average {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let [Value::Features { names, values }] = inputs else {
            return Err("takes one set of features".into());
        };
        // Measures in the order they first appear, with their sum and count.
        let mut measures: Vec<(String, f64, usize)> = Vec::new();
        let mut seen: Vec<&str> = Vec::new();
        for (full, value) in names.iter().zip(values) {
            let Some((channel, measure)) = full.split_once(':') else {
                continue;
            };
            if !self.channels.is_empty() && !self.channels.iter().any(|c| c == channel) {
                continue;
            }
            if !seen.contains(&channel) {
                seen.push(channel);
            }
            match measures.iter_mut().find(|(m, _, _)| m == measure) {
                Some((_, sum, count)) => {
                    *sum += value;
                    *count += 1;
                }
                None => measures.push((measure.to_owned(), *value, 1)),
            }
        }
        if let Some(missing) = self.channels.iter().find(|c| !seen.contains(&c.as_str())) {
            return Err(format!("no features for channel {missing:?}"));
        }
        if measures.is_empty() {
            return Err("no features named <channel>:<measure>".into());
        }
        Ok(Some(Value::Features {
            names: measures.iter().map(|(m, _, _)| format!("{}:{m}", self.name)).collect(),
            values: measures.iter().map(|(_, sum, count)| sum / *count as f64).collect(),
        }))
    }
}

/// `phase_amplitude`: the phase of one band and the amplitude envelope of another in each channel, over the step:
/// the envelope's mean and spread, the unwrapped phase's mean and spread, and how closely the two move together.
struct PhaseAmplitude {
    phase_band: (f64, f64),
    amplitude_band: (f64, f64),
    order: usize,
    trim_s: f64,
    stats: Vec<String>,
}

/// Phase with the jumps of 2π taken out, as NumPy's `unwrap`.
fn unwrap(wrapped: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut out = Vec::with_capacity(wrapped.len());
    let mut shift = 0.0;
    let mut previous = None;
    for &w in wrapped {
        if let Some(p) = previous {
            let d: f64 = w - p;
            let mut m = (d + PI).rem_euclid(2.0 * PI) - PI;
            if m == -PI && d > 0.0 {
                m = PI;
            }
            if d.abs() >= PI {
                shift += m - d;
            }
        }
        out.push(w + shift);
        previous = Some(w);
    }
    out
}

const PHASE_AMPLITUDE_STATS: [&str; 5] = [
    "amplitude_mean",
    "amplitude_sd",
    "phase_mean",
    "phase_sd",
    "correlation",
];

impl Op for PhaseAmplitude {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let slow = Design::bandpass(self.order, self.phase_band.0, self.phase_band.1, signal.rate_hz)
            .map_err(|e| e.to_string())?;
        let fast = Design::bandpass(self.order, self.amplitude_band.0, self.amplitude_band.1, signal.rate_hz)
            .map_err(|e| e.to_string())?;
        let what = format!("{}/{}", label(self.phase_band), label(self.amplitude_band));
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for (channel, x) in signal.channels.iter().zip(&signal.data) {
            let phase = unwrap(&analytic(&filtfilt(&slow, x)).phase);
            let amplitude = analytic(&filtfilt(&fast, x)).amplitude;
            let trim = ((self.trim_s * signal.rate_hz).round() as usize).min(x.len() / 2);
            let (p, a) = (&phase[trim..x.len() - trim], &amplitude[trim..x.len() - trim]);
            let (sp, sa) = (summary(p), summary(a));
            let n = p.len().max(1) as f64;
            let covariance = p.iter().zip(a).map(|(u, v)| (u - sp.mean) * (v - sa.mean)).sum::<f64>() / n;
            for stat in &self.stats {
                names.push(format!("{channel}:phase_amplitude:{what}:{stat}"));
                values.push(match stat.as_str() {
                    "amplitude_mean" => sa.mean,
                    "amplitude_sd" => sa.sd,
                    "phase_mean" => sp.mean,
                    "phase_sd" => sp.sd,
                    _ => covariance / (sp.sd * sa.sd),
                });
            }
        }
        Ok(Some(Value::Features { names, values }))
    }
}

/// A positive setting, or its default.
fn positive(config: &Json, key: &str, default: f64, most: f64) -> Result<f64, String> {
    match config.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_f64()
            .filter(|x| *x > 0.0 && *x <= most)
            .ok_or_else(|| format!("{key} must be a positive number, at most {most}")),
    }
}

/// The channels of a `mix`: `[{"name": ..., "weights": {channel: weight, ...}}, ...]`.
fn mixes(config: &Json) -> Result<Vec<Mixed>, String> {
    let wrong = || "needs channels: a list of {\"name\": ..., \"weights\": {channel: weight}}".to_owned();
    let listed = config.get("channels").and_then(Json::as_array).ok_or_else(wrong)?;
    if listed.is_empty() || listed.len() > MAX_COUNT {
        return Err(wrong());
    }
    let mut channels: Vec<Mixed> = Vec::new();
    for entry in listed {
        let name = entry.get("name").and_then(Json::as_str).ok_or_else(wrong)?;
        let weights = entry.get("weights").and_then(Json::as_object).ok_or_else(wrong)?;
        if channels.iter().any(|(n, _)| n == name) {
            return Err(format!("the channel name {name:?} is used twice"));
        }
        let weights = weights
            .iter()
            .map(|(channel, weight)| {
                weight
                    .as_f64()
                    .filter(|w| w.is_finite())
                    .map(|w| (channel.clone(), w))
                    .ok_or_else(|| format!("the weight of {channel:?} in {name:?} is not a number"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if weights.is_empty() {
            return Err(format!("{name:?} has no weights"));
        }
        channels.push((name.to_owned(), weights));
    }
    Ok(channels)
}

/// The settings `peaks` and `jumps` share, with the defaults of each.
fn seconds(config: &Json, threshold: f64, min_gap: f64) -> Result<Seconds, String> {
    Ok(Seconds {
        threshold: positive(config, "threshold", threshold, 1e6)?,
        back: positive(config, "back_s", 2.0, 60.0)?,
        ahead: positive(config, "ahead_s", 0.25, 60.0)?,
        min_gap: positive(config, "min_gap_s", min_gap, 60.0)?,
        near: positive(config, "near_s", 0.048, 10.0)?,
        far: positive(config, "far_s", 0.152, 10.0)?,
        leak: positive(config, "leak_s", 5.0, 3600.0)?,
        scale_half_life: positive(config, "scale_half_life_s", 120.0, 86400.0)?,
        warmup: positive(config, "warmup_s", 5.0, 600.0)?,
    })
}

/// The block called `op`, with the settings in `config` and numbers from `weights`.
pub(crate) fn build(op: &str, config: &Json, weights: &Weights) -> Result<Box<dyn Op>, String> {
    Ok(match op {
        "select" => Box::new(Select {
            channels: strings(config, "channels", &[])?,
        }),
        "filter" => Box::new(FilterOp {
            steps: config
                .get("steps")
                .and_then(Json::as_array)
                .ok_or("needs steps: a list of filters")?
                .clone(),
            start: start(config, "start")?,
            zero_phase: flag(config, "zero_phase"),
            built: None,
        }),
        "band_power" => {
            let choice = |key: &str, default: &str, allowed: &[&str]| -> Result<String, String> {
                let value = config.get(key).and_then(Json::as_str).unwrap_or(default);
                if allowed.contains(&value) {
                    Ok(value.to_owned())
                } else {
                    Err(format!("{key} must be one of {allowed:?}, not {value:?}"))
                }
            };
            let magnitude = choice("measure", "power", &["power", "magnitude"])? == "magnitude";
            let segment = config.get("segment").map(|_| count(config, "segment", 0)).transpose()?;
            if magnitude && segment.is_some() {
                return Err("segment is for power; magnitude uses the whole step".into());
            }
            Box::new(BandPower {
                bands: bands(config, "bands")?,
                relative_to: config.get("relative_to").map(band).transpose()?,
                log: flag(config, "log"),
                segment,
                magnitude,
                simpson: choice("integrate", "sum", &["sum", "simpson"])? == "simpson",
                hann: choice("window", "hann", &["hann", "none"])? == "hann",
                remove_mean: config.get("remove_mean").and_then(Json::as_bool).unwrap_or(true),
            })
        }
        "ratios" => {
            let pairs: Vec<(String, String)> = config
                .get("pairs")
                .and_then(Json::as_array)
                .ok_or("needs pairs: a list of [measure, measure]")?
                .iter()
                .map(|pair| match pair.as_array().map(Vec::as_slice) {
                    Some([a, b]) => match (a.as_str(), b.as_str()) {
                        (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => Ok((a.to_owned(), b.to_owned())),
                        _ => Err(format!("{pair} is not a pair of measure names")),
                    },
                    _ => Err(format!("{pair} is not a pair of measure names")),
                })
                .collect::<Result<_, _>>()?;
            if pairs.is_empty() || pairs.len() > MAX_COUNT {
                return Err("needs pairs: a list of [measure, measure]".into());
            }
            Box::new(Ratios {
                pairs,
                log: flag(config, "log"),
            })
        }
        "average" => Box::new(Average {
            channels: strings(config, "channels", &[])?,
            // Not `name`, which every node already uses for itself.
            name: match config.get("prefix") {
                None => "mean".to_owned(),
                Some(Json::String(prefix)) if !prefix.is_empty() && !prefix.contains(':') => prefix.clone(),
                Some(other) => return Err(format!("prefix must be a name without ':', not {other}")),
            },
        }),
        "phase_amplitude" => {
            let stats = strings(config, "stats", &PHASE_AMPLITUDE_STATS)?;
            if let Some(unknown) = stats.iter().find(|s| !PHASE_AMPLITUDE_STATS.contains(&s.as_str())) {
                return Err(format!(
                    "unknown statistic {unknown:?}; known: {PHASE_AMPLITUDE_STATS:?}"
                ));
            }
            Box::new(PhaseAmplitude {
                phase_band: band(config.get("phase_band").ok_or("needs phase_band: [low, high]")?)?,
                amplitude_band: band(
                    config
                        .get("amplitude_band")
                        .ok_or("needs amplitude_band: [low, high]")?,
                )?,
                order: count(config, "order", 2)?,
                trim_s: duration(config, "trim_s")?,
                stats,
            })
        }
        "envelope" => Box::new(Envelope {
            band: band(config.get("band").ok_or("needs band: [low, high]")?)?,
            order: count(config, "order", 2)?,
            stats: strings(config, "stats", &["mean", "sd"])?,
        }),
        "summary" => Box::new(Summary {
            stats: strings(config, "stats", &["mean", "sd"])?,
        }),
        "quality" => Box::new(Quality {
            limit: number(config, "limit").unwrap_or(f64::INFINITY),
        }),
        "coherence" => {
            let pairs: Vec<(String, String)> = config
                .get("pairs")
                .and_then(Json::as_array)
                .ok_or("needs pairs: a list of [channel, channel]")?
                .iter()
                .map(|pair| match pair.as_array().map(Vec::as_slice) {
                    Some([a, b]) => match (a.as_str(), b.as_str()) {
                        (Some(a), Some(b)) => Ok((a.to_owned(), b.to_owned())),
                        _ => Err(format!("{pair} is not a pair of channel names")),
                    },
                    _ => Err(format!("{pair} is not a pair of channel names")),
                })
                .collect::<Result<_, _>>()?;
            let steps = count(config, "average_steps", 1)?;
            Box::new(Coherence {
                recent: pairs.iter().map(|_| Recent::new(steps)).collect(),
                pairs,
                bands: bands(config, "bands")?,
                segment: count(config, "segment", 125)?,
            })
        }
        "coupling" => Box::new(Coupling {
            phase_band: band(config.get("phase_band").ok_or("needs phase_band: [low, high]")?)?,
            amplitude_band: band(
                config
                    .get("amplitude_band")
                    .ok_or("needs amplitude_band: [low, high]")?,
            )?,
            order: count(config, "order", 2)?,
            bins: count(config, "bins", 18)?.clamp(2, 360),
            trim_s: duration(config, "trim_s")?,
            steps: count(config, "average_steps", 1)?,
            recent: Vec::new(),
        }),
        "wavelet" => Box::new(Wavelet {
            frequencies: config
                .get("frequencies")
                .and_then(Json::as_array)
                .ok_or("needs frequencies: a list of Hz")?
                .iter()
                .map(|f| {
                    // A wavelet's length grows with cycles over frequency; both are kept sane.
                    f.as_f64()
                        .filter(|f| *f >= 0.1)
                        .ok_or_else(|| format!("{f} is not a frequency of 0.1 Hz or more"))
                })
                .collect::<Result<_, _>>()?,
            cycles: match number(config, "cycles") {
                None => 6.0,
                Some(cycles) if (1.0..=64.0).contains(&cycles) => cycles,
                Some(_) => return Err("cycles must lie between 1 and 64".into()),
            },
            trim_s: duration(config, "trim_s")?,
            log: flag(config, "log"),
        }),
        "heart" => Box::new(HeartOp {
            channel: config.get("channel").and_then(Json::as_str).map(str::to_owned),
            invert: flag(config, "invert"),
            recent: Recent::new(count(config, "window_steps", 1)?),
        }),
        "asr" => {
            let prefix = config
                .get("weights")
                .and_then(Json::as_str)
                .ok_or("needs weights: the stored calibration")?;
            let calibration = AsrCalibration {
                mixing: weights.matrix(&format!("{prefix}.mixing"))?,
                threshold: weights.matrix(&format!("{prefix}.threshold"))?,
            };
            let mut asr = Asr::new(
                &calibration,
                number(config, "max_removed").unwrap_or(0.66),
                number(config, "window_s").unwrap_or(0.5),
            )
            .map_err(|e| e.to_string())?
            .with_average(count(config, "average_windows", 1)?);
            if let Some(weighting) = config.get("weighting") {
                let b = numbers(weighting.get("b").ok_or("weighting needs b and a")?, "weighting.b")?;
                let a = numbers(weighting.get("a").ok_or("weighting needs b and a")?, "weighting.a")?;
                asr = asr.with_weighting(&b, &a).map_err(|e| e.to_string())?;
            }
            match number(config, "blend_offset") {
                None => {}
                Some(offset) if offset == 0.0 || offset == 1.0 => asr = asr.with_blend_offset(offset),
                Some(_) => return Err("blend_offset must be 0 or 1".into()),
            }
            let before = match config.get("before") {
                None if config.get("before_start").is_some() => {
                    return Err("before_start is the start of the before filters; give before too".into());
                }
                None => None,
                Some(steps) => Some(FilterOp {
                    steps: steps.as_array().ok_or("before must be a list of filters")?.clone(),
                    start: start(config, "before_start")?,
                    zero_phase: false,
                    built: None,
                }),
            };
            let passes = count(config, "passes", 1)?;
            if passes > 8 {
                return Err("passes must be at most 8".into());
            }
            Box::new(AsrOp { asr, before, passes })
        }
        "mix" => Box::new(Mix {
            channels: mixes(config)?,
        }),
        "peaks" => Box::new(PeaksOp {
            seconds: seconds(config, 4.0, 0.25)?,
            built: None,
        }),
        "jumps" => {
            let seconds = seconds(config, 6.0, 0.12)?;
            if seconds.near >= seconds.far {
                return Err("near_s must be less than far_s".into());
            }
            Box::new(JumpsOp { seconds, built: None })
        }
        "tilt" => Box::new(Tilt {
            counts_per_g: positive(config, "counts_per_g", 16384.0, 1e9)?,
        }),
        "concat" => Box::new(Concat),
        "normalize" => match config.get("weights").and_then(Json::as_str) {
            Some(prefix) => Box::new(Normalize::Stored {
                center: weights.vector(&format!("{prefix}.center"))?,
                scale: weights.vector(&format!("{prefix}.scale"))?,
            }),
            None => {
                let half_life = number(config, "half_life_steps").ok_or("needs weights, or half_life_steps")?;
                if !(half_life > 0.0 && half_life.is_finite()) {
                    return Err("half_life_steps must be a positive number".into());
                }
                Box::new(Normalize::Running {
                    rate: 1.0 - 0.5f64.powf(1.0 / half_life),
                    warmup: count(config, "warmup_steps", 2)?.max(1),
                    seen: 0,
                    mean: Vec::new(),
                    variance: Vec::new(),
                })
            }
        },
        "history" => Box::new(History {
            recent: Recent::new(count(config, "steps", 1)?),
        }),
        "network" => Box::new(NetworkOp {
            network: Network::build(
                config
                    .get("layers")
                    .and_then(Json::as_array)
                    .ok_or("needs layers: a list")?,
                weights,
            )?,
            names: strings(config, "names", &[])?,
        }),
        other => return Err(format!("unknown op {other:?}")),
    })
}
