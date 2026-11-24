//! The blocks a pipeline is made of.

// Channels and pairs are looked up by indexes found in the same signal just before.
#![allow(clippy::indexing_slicing)]

use serde_json::Value as Json;

use super::nn::{Network, Weights};
use crate::dsp::{
    Asr, AsrCalibration, CouplingSums, CrossSpectrum, Design, Filter, Recent, Start, analytic, cross_spectrum,
    filtfilt, heart, morlet, periodogram, quality, summary, welch,
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

/// `band_power`: power in each band of each channel.
struct BandPower {
    bands: Vec<(f64, f64)>,
    relative_to: Option<(f64, f64)>,
    log: bool,
    segment: Option<usize>,
}

impl Op for BandPower {
    fn reset(&mut self) {}
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for (channel, x) in signal.channels.iter().zip(&signal.data) {
            let spectrum = match self.segment {
                Some(segment) => welch(x, signal.rate_hz, segment),
                None => periodogram(x, signal.rate_hz),
            };
            let total = self.relative_to.map(|(low, high)| spectrum.band_power(low, high));
            for b in &self.bands {
                let power = spectrum.band_power(b.0, b.1) / total.unwrap_or(1.0);
                names.push(format!("{channel}:power:{}", label(*b)));
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

/// `asr`: artifact subspace reconstruction with a stored calibration.
struct AsrOp {
    asr: Asr,
}

impl Op for AsrOp {
    fn reset(&mut self) {
        self.asr.reset();
    }
    fn run(&mut self, inputs: &[&Value]) -> Result<Option<Value>, String> {
        let signal = one_signal(inputs)?;
        Ok(Some(Value::Signal(Signal {
            rate_hz: signal.rate_hz,
            channels: signal.channels.clone(),
            data: self
                .asr
                .process(&signal.data, signal.rate_hz)
                .map_err(|e| e.to_string())?,
        })))
    }
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
            start: match config.get("start").and_then(Json::as_str).unwrap_or("zero") {
                "zero" => Start::Zero,
                "settled" => Start::Settled,
                other => return Err(format!("start must be zero or settled, not {other:?}")),
            },
            zero_phase: flag(config, "zero_phase"),
            built: None,
        }),
        "band_power" => Box::new(BandPower {
            bands: bands(config, "bands")?,
            relative_to: config.get("relative_to").map(band).transpose()?,
            log: flag(config, "log"),
            segment: config.get("segment").map(|_| count(config, "segment", 0)).transpose()?,
        }),
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
            Box::new(AsrOp {
                asr: Asr::new(
                    &calibration,
                    number(config, "max_removed").unwrap_or(0.66),
                    number(config, "window_s").unwrap_or(0.5),
                )
                .map_err(|e| e.to_string())?,
            })
        }
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
