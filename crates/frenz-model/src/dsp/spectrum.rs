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

    /// Power in each band divided by the power between `total.0` and `total.1`.
    pub fn relative_band_power(&self, bands: &[(f64, f64)], total: (f64, f64)) -> Vec<f64> {
        let all = self.band_power(total.0, total.1);
        bands
            .iter()
            .map(|(low, high)| self.band_power(*low, *high) / all)
            .collect()
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
