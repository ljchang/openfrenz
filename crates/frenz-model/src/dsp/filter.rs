//! Recursive filters as cascades of second-order sections.

// Sections and their states are fixed-size arrays, and loops stay inside lengths just checked.
#![allow(clippy::indexing_slicing)]

use rustfft::num_complex::Complex64;

use crate::Error;

/// One second-order section: `b0 + b1 z⁻¹ + b2 z⁻²` over `1 + a1 z⁻¹ + a2 z⁻²`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Section {
    /// Numerator coefficients.
    pub b: [f64; 3],
    /// Denominator coefficients after the leading 1.
    pub a: [f64; 2],
}

impl Section {
    /// Gain for a constant input.
    fn dc_gain(&self) -> f64 {
        (self.b[0] + self.b[1] + self.b[2]) / (1.0 + self.a[0] + self.a[1])
    }

    /// State in which a constant input of 1 gives a constant output.
    fn settled(&self) -> [f64; 2] {
        let gain = self.dc_gain();
        [gain - self.b[0], self.b[2] - self.a[1] * gain]
    }

    fn run(&self, state: &mut [f64; 2], x: f64) -> f64 {
        let y = self.b[0] * x + state[0];
        state[0] = self.b[1] * x - self.a[0] * y + state[1];
        state[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

/// A filter's coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct Design {
    sections: Vec<Section>,
}

fn check(what: &str, ok: bool) -> Result<(), Error> {
    if ok {
        Ok(())
    } else {
        Err(Error::Spec(format!("filter: {what}")))
    }
}

/// Poles of the analog Butterworth low-pass of this order with cut-off 1.
fn prototype(order: usize) -> Vec<Complex64> {
    (0..order)
        .map(|k| {
            let angle = std::f64::consts::PI * (2 * k + order + 1) as f64 / (2 * order) as f64;
            Complex64::from_polar(1.0, angle)
        })
        .collect()
}

/// Analog frequency that the bilinear transform (at a sample rate of 2) maps to `hz`.
fn warp(hz: f64, rate_hz: f64) -> f64 {
    4.0 * (std::f64::consts::PI * hz / rate_hz).tan()
}

/// The digital filter for analog `poles` with `zeros_at_zero` zeros at the origin and analog
/// gain `gain`; its remaining zeros are at infinity.
fn digital(poles: &[Complex64], zeros_at_zero: usize, gain: f64) -> Design {
    let four = Complex64::new(4.0, 0.0);
    let mapped: Vec<Complex64> = poles.iter().map(|p| (four + p) / (four - p)).collect();
    // prod(4 - z) / prod(4 - p): each zero at the origin contributes 4, each one at infinity 1.
    let denominator: Complex64 = poles.iter().map(|p| four - p).product();
    let gain = gain * (Complex64::new(4f64.powi(zeros_at_zero as i32), 0.0) / denominator).re;
    // Zeros at the origin go to +1, zeros at infinity to -1.
    let mut plus = zeros_at_zero;
    let mut minus = poles.len() - zeros_at_zero;
    let mut zero = || {
        if plus > minus {
            plus -= 1;
            1.0
        } else {
            minus -= 1;
            -1.0
        }
    };
    // Conjugate pairs become one section each; real poles are paired among themselves.
    let mut upper: Vec<Complex64> = mapped.iter().copied().filter(|p| p.im > 1e-12).collect();
    let mut real: Vec<f64> = mapped.iter().filter(|p| p.im.abs() <= 1e-12).map(|p| p.re).collect();
    upper.sort_by(|a, b| a.norm().total_cmp(&b.norm()));
    real.sort_by(f64::total_cmp);
    let mut sections = Vec::new();
    for p in upper {
        let (z1, z2) = (zero(), zero());
        sections.push(Section {
            b: [1.0, -(z1 + z2), z1 * z2],
            a: [-2.0 * p.re, p.norm_sqr()],
        });
    }
    for pair in real.chunks(2) {
        sections.push(match pair {
            [p, q] => {
                let (z1, z2) = (zero(), zero());
                Section {
                    b: [1.0, -(z1 + z2), z1 * z2],
                    a: [-(p + q), p * q],
                }
            }
            [p] => Section {
                b: [1.0, -zero(), 0.0],
                a: [-p, 0.0],
            },
            _ => continue,
        });
    }
    if let Some(first) = sections.first_mut() {
        first.b = first.b.map(|b| b * gain);
    }
    Design { sections }
}

impl Design {
    /// A filter from its sections, applied in order.
    pub fn from_sections(sections: Vec<Section>) -> Self {
        Self { sections }
    }

    /// The sections, in the order they are applied.
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// Butterworth low-pass of `order` with its -3 dB point at `high_hz`.
    pub fn lowpass(order: usize, high_hz: f64, rate_hz: f64) -> Result<Self, Error> {
        check("the order must be at least 1", order > 0)?;
        check(
            "the cut-off must lie between 0 and half the sample rate",
            high_hz > 0.0 && high_hz < rate_hz / 2.0,
        )?;
        let w = warp(high_hz, rate_hz);
        let poles: Vec<_> = prototype(order).iter().map(|p| p * w).collect();
        Ok(digital(&poles, 0, w.powi(order as i32)))
    }

    /// Butterworth high-pass of `order` with its -3 dB point at `low_hz`.
    pub fn highpass(order: usize, low_hz: f64, rate_hz: f64) -> Result<Self, Error> {
        check("the order must be at least 1", order > 0)?;
        check(
            "the cut-off must lie between 0 and half the sample rate",
            low_hz > 0.0 && low_hz < rate_hz / 2.0,
        )?;
        let w = warp(low_hz, rate_hz);
        let poles: Vec<_> = prototype(order).iter().map(|p| w / p).collect();
        Ok(digital(&poles, order, 1.0))
    }

    /// Butterworth band-pass with -3 dB points at `low_hz` and `high_hz`. Like SciPy's, a
    /// band-pass of `order` has `2 * order` poles.
    pub fn bandpass(order: usize, low_hz: f64, high_hz: f64, rate_hz: f64) -> Result<Self, Error> {
        check("the order must be at least 1", order > 0)?;
        check(
            "the band must satisfy 0 < low < high < half the sample rate",
            low_hz > 0.0 && low_hz < high_hz && high_hz < rate_hz / 2.0,
        )?;
        let (w1, w2) = (warp(low_hz, rate_hz), warp(high_hz, rate_hz));
        let (width, centre_squared) = (w2 - w1, w1 * w2);
        let mut poles = Vec::with_capacity(2 * order);
        for p in prototype(order) {
            let half = p * (width / 2.0);
            let root = (half * half - centre_squared).sqrt();
            poles.push(half + root);
            poles.push(half - root);
        }
        Ok(digital(&poles, order, width.powi(order as i32)))
    }

    /// Notch at `hz`. `q` is the centre frequency over the width of the notch at -3 dB.
    pub fn notch(hz: f64, q: f64, rate_hz: f64) -> Result<Self, Error> {
        check(
            "the notch must lie between 0 and half the sample rate",
            hz > 0.0 && hz < rate_hz / 2.0,
        )?;
        check("q must be positive", q > 0.0)?;
        let w0 = 2.0 * std::f64::consts::PI * hz / rate_hz;
        let gain = 1.0 / (1.0 + (w0 / q / 2.0).tan());
        Ok(Self::from_sections(vec![Section {
            b: [gain, -2.0 * gain * w0.cos(), gain],
            a: [-2.0 * gain * w0.cos(), 2.0 * gain - 1.0],
        }]))
    }

    /// Removes a constant offset: `y[n] = x[n] - x[n-1] + pole * y[n-1]`. A pole nearer 1
    /// passes lower frequencies and settles more slowly.
    pub fn dc_block(pole: f64) -> Result<Self, Error> {
        check("the pole must lie between 0 and 1", pole > 0.0 && pole < 1.0)?;
        Ok(Self::from_sections(vec![Section {
            b: [1.0, -1.0, 0.0],
            a: [-pole, 0.0],
        }]))
    }

    /// The chain of filters a list of settings describes, as pipelines and the Python bindings
    /// write them. Each step is `{"type": ...}` with its numbers: `lowpass` (`high`, `order`),
    /// `highpass` (`low`, `order`), `bandpass` (`low`, `high`, `order`), `notch` (`hz`, `q`) or
    /// `dc_block` (`pole`). `order` is 2, `q` 30 and `pole` 0.99 unless given.
    pub fn from_config(steps: &[serde_json::Value], rate_hz: f64) -> Result<Self, Error> {
        let mut chain = Self::from_sections(Vec::new());
        for step in steps {
            let kind = step.get("type").and_then(serde_json::Value::as_str).unwrap_or_default();
            let number = |key: &str| step.get(key).and_then(serde_json::Value::as_f64);
            let need = |key: &str| number(key).ok_or_else(|| Error::Spec(format!("filter: {kind} needs {key}")));
            let order = match step.get("order") {
                None => 2,
                Some(order) => order
                    .as_u64()
                    .ok_or_else(|| Error::Spec("filter: order must be a whole number".into()))?
                    as usize,
            };
            chain = chain.then(&match kind {
                "lowpass" => Self::lowpass(order, need("high")?, rate_hz)?,
                "highpass" => Self::highpass(order, need("low")?, rate_hz)?,
                "bandpass" => Self::bandpass(order, need("low")?, need("high")?, rate_hz)?,
                "notch" => Self::notch(need("hz")?, number("q").unwrap_or(30.0), rate_hz)?,
                "dc_block" => Self::dc_block(number("pole").unwrap_or(0.99))?,
                other => return Err(Error::Spec(format!("filter: unknown type {other:?}"))),
            });
        }
        Ok(chain)
    }

    /// This filter followed by `next`.
    #[must_use]
    pub fn then(mut self, next: &Self) -> Self {
        self.sections.extend_from_slice(&next.sections);
        self
    }

    /// The state in which a constant input of 1 passes without a transient.
    fn settled(&self) -> Vec<[f64; 2]> {
        let mut level = 1.0;
        self.sections
            .iter()
            .map(|s| {
                let state = s.settled().map(|v| v * level);
                level *= s.dc_gain();
                state
            })
            .collect()
    }
}

/// What a filter assumes came before its first sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Start {
    /// Silence. The output begins with the filter's response to the signal switching on.
    #[default]
    Zero,
    /// The first sample, held for ever. Avoids the switch-on transient of a signal with an
    /// offset.
    Settled,
}

/// A filter running over one or more channels, keeping its state between calls.
#[derive(Debug, Clone)]
pub struct Filter {
    design: Design,
    start: Start,
    /// Per channel, per section. Empty until the first sample of a channel is seen.
    states: Vec<Vec<[f64; 2]>>,
}

impl Filter {
    /// A filter that has seen no samples.
    pub fn new(design: Design, start: Start) -> Self {
        Self {
            design,
            start,
            states: Vec::new(),
        }
    }

    /// Forgets every sample seen.
    pub fn reset(&mut self) {
        self.states.clear();
    }

    /// Filters the next samples of every channel (`x[channel][sample]`), continuing where the
    /// previous call stopped.
    pub fn process(&mut self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if self.states.len() < x.len() {
            self.states.resize(x.len(), Vec::new());
        }
        x.iter()
            .zip(&mut self.states)
            .map(|(channel, state)| {
                if state.is_empty() {
                    let Some(first) = channel.first() else {
                        return Vec::new();
                    };
                    *state = match self.start {
                        Start::Zero => vec![[0.0; 2]; self.design.sections.len()],
                        Start::Settled => self.design.settled().iter().map(|s| s.map(|v| v * first)).collect(),
                    };
                }
                channel
                    .iter()
                    .map(|sample| {
                        self.design
                            .sections
                            .iter()
                            .zip(state.iter_mut())
                            .fold(*sample, |v, (section, s)| section.run(s, v))
                    })
                    .collect()
            })
            .collect()
    }
}

/// Filters `x` forwards and then backwards, so the result has no delay. Needs the whole signal,
/// so it is for work within one step, not across steps. A signal too short to pad (about six
/// samples per section) is returned unfiltered.
pub fn filtfilt(design: &Design, x: &[f64]) -> Vec<f64> {
    let sections = &design.sections;
    // Samples of padding at each end, as SciPy's `sosfiltfilt` chooses them.
    let zero_b2 = sections.iter().filter(|s| s.b[2] == 0.0).count();
    let zero_a2 = sections.iter().filter(|s| s.a[1] == 0.0).count();
    let pad = 3 * (2 * sections.len() + 1 - zero_b2.min(zero_a2));
    let (Some(first), Some(last)) = (x.first().copied(), x.last().copied()) else {
        return Vec::new();
    };
    if x.len() <= pad {
        return x.to_vec();
    }
    // Odd extension: the signal mirrored through its end points.
    let mut padded = Vec::with_capacity(x.len() + 2 * pad);
    padded.extend((1..=pad).rev().map(|i| 2.0 * first - x[i]));
    padded.extend_from_slice(x);
    padded.extend((1..=pad).map(|i| 2.0 * last - x[x.len() - 1 - i]));
    let settled = design.settled();
    let pass = |signal: &mut Vec<f64>| {
        let Some(start) = signal.first().copied() else { return };
        let mut state: Vec<[f64; 2]> = settled.iter().map(|s| s.map(|v| v * start)).collect();
        for sample in signal.iter_mut() {
            *sample = sections
                .iter()
                .zip(state.iter_mut())
                .fold(*sample, |v, (section, s)| section.run(s, v));
        }
    };
    pass(&mut padded);
    padded.reverse();
    pass(&mut padded);
    padded.reverse();
    padded[pad..pad + x.len()].to_vec()
}
