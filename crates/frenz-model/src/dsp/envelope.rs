//! Amplitude envelope and phase of a signal.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// Instantaneous amplitude and phase of a signal, from its analytic signal (as SciPy's
/// `hilbert`). Meaningful for a signal confined to a narrow band, so filter first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Analytic {
    /// Amplitude envelope of each sample, in signal units.
    pub amplitude: Vec<f64>,
    /// Phase of each sample in radians, between -π and π.
    pub phase: Vec<f64>,
}

/// The envelope and phase of `x`. The first and last few cycles are distorted by the ends of
/// the signal.
pub fn analytic(x: &[f64]) -> Analytic {
    let n = x.len();
    if n == 0 {
        return Analytic::default();
    }
    let mut planner = FftPlanner::new();
    let mut buffer: Vec<Complex64> = x.iter().map(|v| Complex64::new(*v, 0.0)).collect();
    planner.plan_fft_forward(n).process(&mut buffer);
    // Keep the positive frequencies doubled and drop the negative ones.
    let half = n / 2;
    for (k, value) in buffer.iter_mut().enumerate() {
        let keep = if k == 0 || (n & 1 == 0 && k == half) {
            1.0
        } else if k <= (n - 1) / 2 {
            2.0
        } else {
            0.0
        };
        *value *= keep / n as f64;
    }
    planner.plan_fft_inverse(n).process(&mut buffer);
    Analytic {
        amplitude: buffer.iter().map(|v| v.norm()).collect(),
        phase: buffer.iter().map(|v| v.arg()).collect(),
    }
}
