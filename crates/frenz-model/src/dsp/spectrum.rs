//! Power spectra.

// Bins and windows are indexed inside lengths computed just before.
#![allow(clippy::indexing_slicing)]

use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// A one-sided power spectral density: power per hertz at evenly spaced frequencies from 0 to
/// half the sample rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    /// Spacing of the frequencies (Hz); bin `k` is at `k * resolution_hz`.
    pub resolution_hz: f64,
    /// Power per hertz in each bin, in squared signal units per hertz.
    pub density: Vec<f64>,
}

impl Spectrum {
    /// Frequency of each bin (Hz).
    pub fn frequencies(&self) -> Vec<f64> {
        (0..self.density.len()).map(|k| k as f64 * self.resolution_hz).collect()
    }

    /// Power between `low_hz` (included) and `high_hz` (excluded): the bins' density times
    /// their width, summed. In squared signal units.
    pub fn band_power(&self, low_hz: f64, high_hz: f64) -> f64 {
        self.density
            .iter()
            .enumerate()
            .filter(|(k, _)| {
                let f = *k as f64 * self.resolution_hz;
                f >= low_hz && f < high_hz
            })
            .map(|(_, p)| p * self.resolution_hz)
            .sum()
    }

    /// The band between `low_hz` and `high_hz` integrated with Simpson's rule over the bins nearest the two edges,
    /// both included. Fewer than two bins give the single bin's value times the resolution; an even number of
    /// bins uses Simpson's rule on all but the last interval and the trapezoid rule on that one.
    pub fn band_simpson(&self, low_hz: f64, high_hz: f64) -> f64 {
        if self.density.is_empty() || self.resolution_hz <= 0.0 {
            return f64::NAN;
        }
        let last = self.density.len() - 1;
        let bin = |hz: f64| ((hz / self.resolution_hz).round().max(0.0) as usize).min(last);
        let (from, to) = (bin(low_hz), bin(high_hz));
        if to <= from {
            return self.density[from] * self.resolution_hz;
        }
        simpson(&self.density[from..=to], self.resolution_hz)
    }

    /// Power in each band divided by the power between `total.0` and `total.1`.
    pub fn relative_band_power(&self, bands: &[(f64, f64)], total: (f64, f64)) -> Vec<f64> {
        let all = self.band_power(total.0, total.1);
        bands
            .iter()
            .map(|(low, high)| self.band_power(*low, *high) / all)
            .collect()
    }
}

/// Composite Simpson's rule on evenly spaced values `y`, spacing `h`. An even number of values uses Simpson's
/// rule on all but the last interval and the trapezoid rule on that one.
fn simpson(y: &[f64], h: f64) -> f64 {
    match y.len() {
        0 => 0.0,
        1 => y[0] * h,
        2 => h * (y[0] + y[1]) / 2.0,
        n if n % 2 == 1 => {
            let inner: f64 = y[1..n - 1]
                .iter()
                .enumerate()
                .map(|(i, v)| if i % 2 == 0 { 4.0 * v } else { 2.0 * v })
                .sum();
            h / 3.0 * (y[0] + inner + y[n - 1])
        }
        n => simpson(&y[..n - 1], h) + h * (y[n - 2] + y[n - 1]) / 2.0,
    }
}

/// The size of each frequency in the whole of `x`: the magnitude of its Fourier transform at frequencies from
/// 0 to half the sample rate, unscaled. With `hann`, a Hann window first; with `remove_mean`, the mean is
/// subtracted first. Useful for shares of a total, where the scale cancels.
pub fn magnitude(x: &[f64], rate_hz: f64, hann_window: bool, remove_mean: bool) -> Spectrum {
    let n = x.len();
    if n == 0 {
        return Spectrum {
            resolution_hz: 0.0,
            density: Vec::new(),
        };
    }
    let mean = if remove_mean {
        x.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };
    let window = if hann_window { hann(n) } else { vec![1.0; n] };
    let mut buffer: Vec<Complex64> = x
        .iter()
        .zip(&window)
        .map(|(v, w)| Complex64::new((v - mean) * w, 0.0))
        .collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut buffer);
    Spectrum {
        resolution_hz: rate_hz / n as f64,
        density: buffer[..n / 2 + 1].iter().map(|c| c.norm()).collect(),
    }
}

fn hann(n: usize) -> Vec<f64> {
    // The periodic form, as spectral estimates use.
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

/// Density of one stretch of signal: mean removed, Hann window.
fn segment_density(x: &[f64], rate_hz: f64, planner: &mut FftPlanner<f64>) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let window = hann(n);
    let mut buffer: Vec<Complex64> = x
        .iter()
        .zip(&window)
        .map(|(v, w)| Complex64::new((v - mean) * w, 0.0))
        .collect();
    planner.plan_fft_forward(n).process(&mut buffer);
    let scale = 1.0 / (rate_hz * window.iter().map(|w| w * w).sum::<f64>());
    let bins = n / 2 + 1;
    (0..bins)
        .map(|k| {
            // Every frequency but 0 and (for an even length) the highest appears twice.
            let twice = k > 0 && !(n & 1 == 0 && k == bins - 1);
            buffer[k].norm_sqr() * scale * if twice { 2.0 } else { 1.0 }
        })
        .collect()
}

/// Power spectral density of the whole of `x`, with its mean removed and a Hann window. The
/// resolution is the sample rate over the length. An empty signal gives an empty spectrum.
pub fn periodogram(x: &[f64], rate_hz: f64) -> Spectrum {
    if x.is_empty() {
        return Spectrum {
            resolution_hz: 0.0,
            density: Vec::new(),
        };
    }
    Spectrum {
        resolution_hz: rate_hz / x.len() as f64,
        density: segment_density(x, rate_hz, &mut FftPlanner::new()),
    }
}

/// Power spectral density averaged over half-overlapping stretches of `segment` samples
/// (Welch's method): steadier than [`periodogram`], at a resolution of the sample rate over
/// `segment`. A signal shorter than `segment` is treated as a single stretch.
pub fn welch(x: &[f64], rate_hz: f64, segment: usize) -> Spectrum {
    let segment = segment.min(x.len());
    if segment == 0 {
        return periodogram(x, rate_hz);
    }
    let hop = (segment - segment / 2).max(1);
    let mut planner = FftPlanner::new();
    let mut density = vec![0.0; segment / 2 + 1];
    let mut count = 0.0;
    let mut from = 0;
    while from + segment <= x.len() {
        for (sum, p) in density
            .iter_mut()
            .zip(segment_density(&x[from..from + segment], rate_hz, &mut planner))
        {
            *sum += p;
        }
        count += 1.0;
        from += hop;
    }
    density.iter_mut().for_each(|p| *p /= count);
    Spectrum {
        resolution_hz: rate_hz / segment as f64,
        density,
    }
}
