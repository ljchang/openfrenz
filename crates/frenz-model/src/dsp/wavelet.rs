//! Time-frequency analysis with Morlet wavelets.

// The kernel is indexed around its centre, inside its own length.
#![allow(clippy::indexing_slicing)]

use rustfft::num_complex::Complex64;

use super::envelope::Analytic;

/// Amplitude and phase of `x` at `hz`, sample by sample, from a Morlet wavelet spanning `cycles`
/// cycles (5 to 7 is usual: more cycles separate frequencies better and smear time more).
///
/// The amplitude is in signal units: a sine of amplitude 3 at `hz` reads 3. The mean of `x` is
/// removed first. The first and last `cycles / hz` seconds or so see the ends of the signal and
/// read low.
pub fn morlet(x: &[f64], rate_hz: f64, hz: f64, cycles: f64) -> Analytic {
    if x.is_empty() || hz <= 0.0 || cycles <= 0.0 || rate_hz <= 0.0 {
        return Analytic::default();
    }
    // A Gaussian of this many seconds' spread, times a complex sine.
    let spread = cycles / (2.0 * std::f64::consts::PI * hz);
    let half = (4.0 * spread * rate_hz).ceil() as usize;
    let gauss: Vec<f64> = (0..=2 * half)
        .map(|k| {
            let t = (k as f64 - half as f64) / rate_hz;
            (-t * t / (2.0 * spread * spread)).exp()
        })
        .collect();
    let total: f64 = gauss.iter().sum();
    let kernel: Vec<Complex64> = gauss
        .iter()
        .enumerate()
        .map(|(k, g)| {
            let t = (k as f64 - half as f64) / rate_hz;
            Complex64::from_polar(2.0 * g / total, 2.0 * std::f64::consts::PI * hz * t)
        })
        .collect();
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let n = x.len();
    let (mut amplitude, mut phase) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        // out[i] = sum over k of x[i - (k - half)] * kernel[k], with nothing beyond the ends.
        let first = (i + half).saturating_sub(n - 1);
        let last = (i + half).min(2 * half);
        let mut sum = Complex64::default();
        for k in first..=last {
            sum += kernel[k] * (x[i + half - k] - mean);
        }
        amplitude.push(sum.norm());
        phase.push(sum.arg());
    }
    Analytic { amplitude, phase }
}

/// Amplitude of `x` at each of `frequencies` over time: `[frequency][sample]`. See [`morlet`].
pub fn scalogram(x: &[f64], rate_hz: f64, frequencies: &[f64], cycles: f64) -> Vec<Vec<f64>> {
    frequencies
        .iter()
        .map(|hz| morlet(x, rate_hz, *hz, cycles).amplitude)
        .collect()
}
