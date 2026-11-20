//! Estimating sample acquisition times from packet arrival times.
//!
//! A packet carrying samples `n .. n+k` cannot arrive before its last sample was acquired, and
//! Bluetooth only adds delay. So, for a stream sampled at a steady rate, the true clock
//! `t(n) = a + b·n` is the *lower envelope* of `(index of last sample, arrival time)`: arrivals lie
//! on or above it. [`fit_lower_envelope`] finds that line by iteratively refitting least squares to
//! the earliest-arriving packets. `1/b` is the measured sample rate; residuals are the delivery
//! latency beyond the minimum (jitter).

use serde_json::json;

/// A fitted clock for one uninterrupted segment of one stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockFit {
    /// Time of sample index 0 (Unix seconds, host clock).
    pub offset: f64,
    /// Seconds per sample.
    pub period: f64,
    /// Median arrival delay above the envelope (s).
    pub jitter_p50: f64,
    /// 95th-percentile arrival delay above the envelope (s).
    pub jitter_p95: f64,
    /// Packets used.
    pub packets: usize,
    /// True if there were too few packets to fit and the nominal rate was used.
    pub nominal: bool,
}

impl ClockFit {
    /// Estimated acquisition time of sample `n`.
    pub fn time(&self, n: f64) -> f64 {
        self.offset + self.period * n
    }

    /// JSON description for session metadata.
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "rate_hz": 1.0 / self.period, "offset_unix_s": self.offset, "packets": self.packets,
            "jitter_p50_s": self.jitter_p50, "jitter_p95_s": self.jitter_p95, "nominal_rate_used": self.nominal,
        })
    }
}

fn least_squares(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = points.len() as f64;
    if points.len() < 2 {
        return None;
    }
    let (sx, sy) = points.iter().fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let (sxx, sxy) = points.iter().fold((0.0, 0.0), |(a, b), (x, y)| {
        (a + (x - mx).powi(2), b + (x - mx) * (y - my))
    });
    if sxx <= 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    Some((my - slope * mx, slope))
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted.get(idx).copied().unwrap_or(0.0)
}

/// Fits the lower envelope of `(last_sample_index, arrival_unix_s)` pairs. With fewer than 20
/// packets, the nominal period is used and only the offset is fitted.
pub fn fit_lower_envelope(points: &[(f64, f64)], nominal_rate_hz: f64) -> ClockFit {
    let nominal_period = 1.0 / nominal_rate_hz;
    let (mut offset, mut period, nominal) = if points.len() >= 20 {
        let mut fit = least_squares(points).unwrap_or((0.0, nominal_period));
        for _ in 0..4 {
            let mut residuals: Vec<f64> = points.iter().map(|(x, y)| y - (fit.0 + fit.1 * x)).collect();
            residuals.sort_by(f64::total_cmp);
            let cut = percentile(&residuals, 0.10);
            let low: Vec<(f64, f64)> = points
                .iter()
                .copied()
                .filter(|(x, y)| y - (fit.0 + fit.1 * x) <= cut)
                .collect();
            match least_squares(&low) {
                // Reject fits that are physically implausible (> 5 % off nominal).
                Some(f) if (f.1 / nominal_period - 1.0).abs() < 0.05 => fit = f,
                _ => break,
            }
        }
        (fit.0, fit.1, false)
    } else {
        (0.0, nominal_period, true)
    };
    // Shift the line down so no packet arrives before its last sample was acquired.
    let min_residual = points
        .iter()
        .map(|(x, y)| y - (offset + period * x))
        .fold(f64::INFINITY, f64::min);
    if min_residual.is_finite() {
        offset += min_residual;
    }
    if !period.is_finite() || period <= 0.0 {
        period = nominal_period;
    }
    let mut delays: Vec<f64> = points.iter().map(|(x, y)| y - (offset + period * x)).collect();
    delays.sort_by(f64::total_cmp);
    ClockFit {
        offset,
        period,
        jitter_p50: percentile(&delays, 0.5),
        jitter_p95: percentile(&delays, 0.95),
        packets: points.len(),
        nominal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_rate_and_offset_under_one_sided_jitter() {
        // 125.3 Hz true rate, 9 samples per packet, delays 5–150 ms (deterministic pseudo-random).
        let true_period = 1.0 / 125.3;
        let mut state = 7u64;
        let points: Vec<(f64, f64)> = (0..2000)
            .map(|k| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let delay = 0.005 + 0.145 * ((state >> 33) as f64 / (1u64 << 31) as f64).powi(3);
                let n = (9 * k + 8) as f64;
                (n, 1_800_000_000.0 + true_period * n + delay)
            })
            .collect();
        let fit = fit_lower_envelope(&points, 125.0);
        assert!((1.0 / fit.period - 125.3).abs() < 0.01, "rate {}", 1.0 / fit.period);
        assert!(
            (fit.offset - 1_800_000_000.005).abs() < 0.003,
            "offset error {}",
            fit.offset - 1_800_000_000.005
        );
        assert!(points.iter().all(|(x, y)| *y >= fit.time(*x) - 1e-9));
    }

    #[test]
    fn short_segments_use_nominal_rate() {
        let points = [(8.0, 10.1), (17.0, 10.2)];
        let fit = fit_lower_envelope(&points, 125.0);
        assert!(fit.nominal);
        assert!((fit.period - 0.008).abs() < 1e-12);
    }
}
