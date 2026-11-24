//! How two signals relate: coherence between channels and coupling between rhythms.

// Bins are indexed inside lengths computed just before.
#![allow(clippy::indexing_slicing)]

use std::collections::VecDeque;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// The spectra of two signals and their cross-spectrum, summed over stretches of signal.
///
/// Coherence from a single stretch is always 1, and from a few it is biased upwards, so these
/// sums are made to be added up: over the stretches of one step ([`cross_spectrum`]) and over
/// several steps ([`CrossSpectrum::add`], [`Recent`]).
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSpectrum {
    /// Spacing of the frequencies (Hz); bin `k` is at `k * resolution_hz`.
    pub resolution_hz: f64,
    /// Stretches summed so far.
    pub stretches: usize,
    xx: Vec<f64>,
    yy: Vec<f64>,
    xy: Vec<Complex64>,
}

fn windowed(x: &[f64], planner: &mut FftPlanner<f64>) -> Vec<Complex64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut buffer: Vec<Complex64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
            Complex64::new((v - mean) * window, 0.0)
        })
        .collect();
    planner.plan_fft_forward(n).process(&mut buffer);
    buffer.truncate(n / 2 + 1);
    buffer
}

/// Spectra of `x` and `y` summed over half-overlapping stretches of `segment` samples, each
/// with its mean removed and a Hann window. The signals are cut to the shorter of the two; if
/// that is shorter than `segment`, it is one stretch.
pub fn cross_spectrum(x: &[f64], y: &[f64], rate_hz: f64, segment: usize) -> CrossSpectrum {
    let n = x.len().min(y.len());
    let segment = segment.min(n);
    let bins = if segment == 0 { 0 } else { segment / 2 + 1 };
    let mut sum = CrossSpectrum {
        resolution_hz: if segment == 0 { 0.0 } else { rate_hz / segment as f64 },
        stretches: 0,
        xx: vec![0.0; bins],
        yy: vec![0.0; bins],
        xy: vec![Complex64::default(); bins],
    };
    if segment == 0 {
        return sum;
    }
    let hop = (segment - segment / 2).max(1);
    let mut planner = FftPlanner::new();
    let mut from = 0;
    while from + segment <= n {
        let fx = windowed(&x[from..from + segment], &mut planner);
        let fy = windowed(&y[from..from + segment], &mut planner);
        for k in 0..bins {
            sum.xx[k] += fx[k].norm_sqr();
            sum.yy[k] += fy[k].norm_sqr();
            sum.xy[k] += fx[k].conj() * fy[k];
        }
        sum.stretches += 1;
        from += hop;
    }
    sum
}

impl CrossSpectrum {
    /// Adds the stretches of `other`. Sums with a different resolution are left out.
    pub fn add(&mut self, other: &Self) {
        if self.xx.is_empty() {
            *self = other.clone();
        } else if self.xx.len() == other.xx.len() && self.resolution_hz == other.resolution_hz {
            for k in 0..self.xx.len() {
                self.xx[k] += other.xx[k];
                self.yy[k] += other.yy[k];
                self.xy[k] += other.xy[k];
            }
            self.stretches += other.stretches;
        }
    }

    /// Magnitude-squared coherence in each bin: 0 for unrelated signals, 1 for a fixed linear
    /// relation. Only meaningful over several stretches.
    pub fn coherence(&self) -> Vec<f64> {
        (0..self.xx.len())
            .map(|k| {
                let power = self.xx[k] * self.yy[k];
                if power > 0.0 {
                    self.xy[k].norm_sqr() / power
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Coherence of the band from `low_hz` (included) to `high_hz` (excluded), taking the band
    /// as a whole: the spectra are summed over its bins first. `NaN` if the band holds no bin.
    pub fn band_coherence(&self, low_hz: f64, high_hz: f64) -> f64 {
        let (mut xx, mut yy, mut xy) = (0.0, 0.0, Complex64::default());
        let mut any = false;
        for k in 0..self.xx.len() {
            let f = k as f64 * self.resolution_hz;
            if f >= low_hz && f < high_hz {
                xx += self.xx[k];
                yy += self.yy[k];
                xy += self.xy[k];
                any = true;
            }
        }
        match (any, xx * yy > 0.0) {
            (false, _) => f64::NAN,
            (true, false) => 0.0,
            (true, true) => xy.norm_sqr() / (xx * yy),
        }
    }

    /// How far `y` lags `x` in the band, as a phase in radians between -π and π (positive: `y`
    /// is later).
    pub fn band_phase_lag(&self, low_hz: f64, high_hz: f64) -> f64 {
        let xy: Complex64 = (0..self.xy.len())
            .filter(|k| {
                let f = *k as f64 * self.resolution_hz;
                f >= low_hz && f < high_hz
            })
            .map(|k| self.xy[k])
            .sum();
        -xy.arg()
    }
}

/// The last few steps' worth of something that is summed over steps.
#[derive(Debug, Clone)]
pub struct Recent<T> {
    steps: usize,
    kept: VecDeque<T>,
}

impl<T> Recent<T> {
    /// Keeps the newest `steps` items (at least one).
    pub fn new(steps: usize) -> Self {
        Self {
            steps: steps.max(1),
            kept: VecDeque::new(),
        }
    }

    /// Adds the newest item, dropping the oldest once `steps` are held.
    pub fn push(&mut self, item: T) {
        if self.kept.len() == self.steps {
            self.kept.pop_front();
        }
        self.kept.push_back(item);
    }

    /// Whether as many items are held as were asked for.
    pub fn is_full(&self) -> bool {
        self.kept.len() == self.steps
    }

    /// The items held, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.kept.iter()
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        self.kept.clear();
    }
}

impl Recent<CrossSpectrum> {
    /// The sum over the steps held; `None` if none are.
    pub fn total(&self) -> Option<CrossSpectrum> {
        let mut steps = self.kept.iter();
        let mut sum = steps.next()?.clone();
        steps.for_each(|s| sum.add(s));
        Some(sum)
    }
}

/// Pearson correlation of `a` and `b`, cut to the shorter. 0 if either does not vary.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let (ma, mb) = (a.iter().sum::<f64>() / n as f64, b.iter().sum::<f64>() / n as f64);
    let (mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0);
    for (p, q) in a.iter().zip(b) {
        aa += (p - ma).powi(2);
        bb += (q - mb).powi(2);
        ab += (p - ma) * (q - mb);
    }
    if aa * bb > 0.0 { ab / (aa * bb).sqrt() } else { 0.0 }
}

/// How strongly the size of one rhythm follows the phase of another.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Coupling {
    /// Modulation index (Tort et al. 2010): how far the amplitude's spread over the phases is
    /// from even. 0 for none; values of a few hundredths are already strong coupling.
    pub modulation_index: f64,
    /// Length of the amplitude-weighted mean phase vector, scaled to lie between 0 and 1
    /// whatever the amplitude (Özkurt & Schnitzler 2011).
    pub mean_vector_length: f64,
    /// The phase at which the amplitude is largest, in radians.
    pub preferred_phase: f64,
}

/// The sums the coupling measures are made from, so they can be added up over several steps
/// ([`CouplingSums::add`], [`Recent`]) when one step is too short to measure coupling in.
#[derive(Debug, Clone, PartialEq)]
pub struct CouplingSums {
    amplitude: Vec<f64>,
    count: Vec<f64>,
    vector: Complex64,
    squares: f64,
    samples: f64,
}

impl CouplingSums {
    /// Sums over a `phase` (radians, one per sample) and an `amplitude` at the same samples,
    /// with the phases sorted into `bins` equal bins (18 is customary; at least 2).
    pub fn new(phase: &[f64], amplitude: &[f64], bins: usize) -> Self {
        let n = phase.len().min(amplitude.len());
        let bins = bins.max(2);
        let mut sums = Self {
            amplitude: vec![0.0; bins],
            count: vec![0.0; bins],
            vector: Complex64::default(),
            squares: 0.0,
            samples: n as f64,
        };
        let width = 2.0 * std::f64::consts::PI / bins as f64;
        for (p, a) in phase[..n].iter().zip(&amplitude[..n]) {
            let bin = (((p + std::f64::consts::PI) / width).floor().max(0.0) as usize).min(bins - 1);
            sums.amplitude[bin] += a;
            sums.count[bin] += 1.0;
            sums.vector += Complex64::from_polar(*a, *p);
            sums.squares += a * a;
        }
        sums
    }

    /// Adds the samples of `other`. Sums with a different number of bins are left out.
    pub fn add(&mut self, other: &Self) {
        if self.amplitude.len() != other.amplitude.len() {
            return;
        }
        for k in 0..self.amplitude.len() {
            self.amplitude[k] += other.amplitude[k];
            self.count[k] += other.count[k];
        }
        self.vector += other.vector;
        self.squares += other.squares;
        self.samples += other.samples;
    }

    /// The coupling measures of everything summed.
    pub fn measure(&self) -> Coupling {
        let bins = self.amplitude.len() as f64;
        let means: Vec<f64> = self
            .amplitude
            .iter()
            .zip(&self.count)
            .map(|(s, c)| if *c > 0.0 { s / c } else { 0.0 })
            .collect();
        let total: f64 = means.iter().sum();
        let modulation_index = if total > 0.0 {
            let entropy: f64 = means
                .iter()
                .filter(|m| **m > 0.0)
                .map(|m| -(m / total) * (m / total).ln())
                .sum();
            (bins.ln() - entropy) / bins.ln()
        } else {
            0.0
        };
        let scale = self.samples * self.squares;
        Coupling {
            modulation_index,
            mean_vector_length: if scale > 0.0 {
                self.vector.norm() / scale.sqrt()
            } else {
                0.0
            },
            preferred_phase: self.vector.arg(),
        }
    }
}

impl Recent<CouplingSums> {
    /// The sum over the steps held; `None` if none are.
    pub fn total(&self) -> Option<CouplingSums> {
        let mut steps = self.kept.iter();
        let mut sum = steps.next()?.clone();
        steps.for_each(|s| sum.add(s));
        Some(sum)
    }
}

/// Coupling between a `phase` (radians, one per sample) and an `amplitude` at the same samples,
/// with the phases sorted into `bins` equal bins for the modulation index (18 is customary).
///
/// Take the phase from a slow band and the amplitude from a faster one with
/// [`crate::dsp::analytic`], and leave out the ends of the signal, where both are distorted.
/// Both measures need many cycles of the slow rhythm to mean anything: a few seconds at least.
pub fn coupling(phase: &[f64], amplitude: &[f64], bins: usize) -> Coupling {
    CouplingSums::new(phase, amplitude, bins).measure()
}
