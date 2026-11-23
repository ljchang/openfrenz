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

/// Where a packet's samples sit within a stream: which uninterrupted segment, and the index of
/// its first sample in that segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placed {
    /// Segment number, from 0.
    pub segment: i64,
    /// Index within the segment of the packet's first sample.
    pub first_sample: i64,
}

#[derive(Debug, Clone, Copy)]
struct Previous {
    arrival: f64,
    device_time: u32,
    sequence: Option<u8>,
    samples: usize,
}

/// Assigns packets of one stream to segments and sample indices, one packet at a time.
///
/// The rules use only the packets seen so far, so the exporter and live readers place packets
/// identically. A new segment starts after a silence of more than a second, after a jump in the
/// sequence counter that the elapsed time does not explain, and, for streams without a counter,
/// after a step in the band's clock or a burst of packets.
#[derive(Debug, Clone)]
pub struct Segmenter {
    rate_hz: f64,
    packet_period: f64,
    per_second: usize,
    previous: Option<Previous>,
    segment: i64,
    next: i64,
    same_second: usize,
    /// Frames missing inside segments (counted from sequence numbers).
    pub lost_frames: u64,
    /// Sequence counter restarts that followed a stream-start command.
    pub restarts: u64,
}

impl Segmenter {
    /// A segmenter for a stream sampled at `rate_hz`.
    pub fn new(rate_hz: f64) -> Self {
        Self {
            rate_hz,
            packet_period: 0.0,
            per_second: 0,
            previous: None,
            segment: 0,
            next: 0,
            same_second: 0,
            lost_frames: 0,
            restarts: 0,
        }
    }

    /// Places the next packet. `arrival` is in seconds, `sequence` is the frame counter if the
    /// stream has one, `samples` the number of samples in the packet. `forced` starts a new
    /// segment regardless (an undecodable frame came before this one). `after_start` says a
    /// stream-start command was written within the last second, which explains a counter restart.
    pub fn push(
        &mut self,
        arrival: f64,
        device_time: u32,
        sequence: Option<u8>,
        samples: usize,
        forced: bool,
        after_start: bool,
    ) -> Placed {
        let Some(prev) = self.previous else {
            self.packet_period = samples as f64 / self.rate_hz;
            // Sequence-less streams send a fixed number of packets per device second (2 observed);
            // more means a burst (seen during PPG freezes), after which the sample count no longer
            // tracks time.
            self.per_second = (1.0 / self.packet_period.max(1e-9)).round() as usize;
            self.same_second = 1;
            self.previous = Some(Previous {
                arrival,
                device_time,
                sequence,
                samples,
            });
            return Placed {
                segment: self.segment,
                first_sample: self.next,
            };
        };
        let packet_period = self.packet_period;
        let step = match (prev.sequence, sequence) {
            (Some(a), Some(b)) => {
                let d = i64::from(b.wrapping_sub(a));
                if d == 1 {
                    Some(1)
                } else if b == 0 && after_start {
                    // Continuous only if data kept flowing; after a pause it is a new segment.
                    self.restarts += 1;
                    (arrival - prev.arrival < 4.0 * packet_period + 0.5).then_some(1)
                } else if d > 1 && arrival - prev.arrival < (d as f64 + 4.0) * packet_period + 1.0 {
                    self.lost_frames += (d - 1) as u64;
                    Some(d)
                } else {
                    None
                }
            }
            // Streams without a sequence counter: a long silence or a device clock jump starts a
            // new segment, because lost packets cannot be counted.
            _ => {
                let gap = arrival - prev.arrival;
                let clock_step = i64::from(device_time) - i64::from(prev.device_time);
                self.same_second = if clock_step == 0 { self.same_second + 1 } else { 1 };
                let burst = self.same_second > self.per_second;
                (gap < 2.0 * packet_period.max(0.5) && (0..=1).contains(&clock_step) && !burst).then_some(1)
            }
        };
        // Any silence of more than a second starts a new segment, whatever the counters say.
        let paused = arrival - prev.arrival > 1.0 + 2.0 * packet_period;
        match step {
            Some(d) if !forced && !paused => self.next += d * prev.samples as i64,
            _ => {
                self.segment += 1;
                self.next = 0;
                self.same_second = 1;
            }
        }
        self.previous = Some(Previous {
            arrival,
            device_time,
            sequence,
            samples,
        });
        Placed {
            segment: self.segment,
            first_sample: self.next,
        }
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
    let (offset, period, nominal) = if points.len() >= 20 {
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
    place_on_envelope(points, offset, period, nominal_period, nominal)
}

/// Fits only the offset, taking the sample rate as `nominal_rate_hz`. For when there is too
/// little data to trust a fitted rate.
pub fn fit_nominal(points: &[(f64, f64)], nominal_rate_hz: f64) -> ClockFit {
    let nominal_period = 1.0 / nominal_rate_hz;
    place_on_envelope(points, 0.0, nominal_period, nominal_period, true)
}

fn place_on_envelope(
    points: &[(f64, f64)],
    mut offset: f64,
    mut period: f64,
    nominal_period: f64,
    nominal: bool,
) -> ClockFit {
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
