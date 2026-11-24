//! Heartbeats from a pulse (PPG) signal.

// Peaks are indexed inside the signal they were found in.
#![allow(clippy::indexing_slicing)]

use super::filter::{Design, filtfilt};

/// Heartbeats found in a stretch of pulse signal, and what they say.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Heart {
    /// Time of each beat, in seconds from the start of the stretch.
    pub beats: Vec<f64>,
    /// Heart rate in beats per minute, from the mean interval between beats. `NaN` when fewer
    /// than two trustworthy intervals were found.
    pub rate_bpm: f64,
    /// Standard deviation of the intervals between beats, in milliseconds (SDNN).
    pub sdnn_ms: f64,
    /// Root mean square of the change from one interval to the next, in milliseconds (RMSSD).
    pub rmssd_ms: f64,
    /// The share of intervals that look like heartbeats: between 0.3 and 1.5 s, within 30% of
    /// the median, and between two beats of ordinary size (a third to three times the median).
    /// Near 1 for a clean signal; treat everything else as unreliable below about 0.8.
    pub quality: f64,
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    match sorted.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => sorted[n / 2],
        n => 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]),
    }
}

/// Finds the beats in `ppg`, sampled at `rate_hz`, taking each pulse as a rise in the signal
/// (negate the signal first if its pulses point down).
///
/// The signal is band-passed to 0.7–3.5 Hz (42 to 210 beats a minute), its peaks are found and
/// their times refined between samples. Interval statistics use only the intervals that pass
/// the checks described at [`Heart::quality`]. Variability measures need a long stretch (a
/// minute or more is usual for SDNN) and a slow sample rate limits them: at 25 Hz each beat is
/// timed to a few milliseconds at best, which adds that much to both measures.
pub fn heart(ppg: &[f64], rate_hz: f64) -> Heart {
    let nothing = Heart {
        rate_bpm: f64::NAN,
        sdnn_ms: f64::NAN,
        rmssd_ms: f64::NAN,
        ..Heart::default()
    };
    let Ok(design) = Design::bandpass(2, 0.7, 3.5, rate_hz) else {
        return nothing;
    };
    let x = filtfilt(&design, ppg);
    if x.len() < 3 {
        return nothing;
    }
    // Peaks standing well above the usual size of the signal around them. Judging each peak
    // against its own few seconds keeps one large disturbance from hiding the beats elsewhere.
    let reach = (3.0 * rate_hz).ceil() as usize;
    let block = reach.max(1);
    let usual: Vec<f64> = (0..x.len().div_ceil(block))
        .map(|b| {
            let from = (b * block).saturating_sub(reach);
            let to = ((b + 1) * block + reach).min(x.len());
            let mut sizes: Vec<f64> = x[from..to].iter().map(|v| v.abs()).collect();
            sizes.sort_by(f64::total_cmp);
            sizes[(0.9 * (sizes.len() - 1) as f64) as usize]
        })
        .collect();
    let mut peaks: Vec<usize> = (1..x.len() - 1)
        .filter(|i| x[*i] > x[i - 1] && x[*i] >= x[i + 1] && x[*i] > 0.3 * usual[i / block] && x[*i] > 0.0)
        .collect();
    // No two beats within 0.3 s: of two close peaks the larger is the beat.
    let apart = (0.3 * rate_hz).ceil() as usize;
    let mut kept: Vec<usize> = Vec::new();
    for peak in peaks.drain(..) {
        match kept.last().copied() {
            Some(previous) if peak - previous < apart => {
                if x[peak] > x[previous] {
                    kept.pop();
                    kept.push(peak);
                }
            }
            _ => kept.push(peak),
        }
    }
    // The top of a parabola through the peak and its neighbours, for a time between samples.
    let beats: Vec<f64> = kept
        .iter()
        .map(|i| {
            let (a, b, c) = (x[i - 1], x[*i], x[i + 1]);
            let bend = a - 2.0 * b + c;
            let shift = if bend < 0.0 { 0.5 * (a - c) / bend } else { 0.0 };
            (*i as f64 + shift) / rate_hz
        })
        .collect();
    let intervals: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    if intervals.is_empty() {
        return Heart { beats, ..nothing };
    }
    let middle = median(&intervals);
    // A beat far larger or smaller than most is not a beat: the pulse changes size slowly.
    let heights: Vec<f64> = kept.iter().map(|i| x[*i]).collect();
    let height = median(&heights);
    let sized: Vec<bool> = heights
        .iter()
        .map(|h| *h >= height / 3.0 && *h <= 3.0 * height)
        .collect();
    let good: Vec<bool> = intervals
        .iter()
        .enumerate()
        .map(|(k, i)| (0.3..=1.5).contains(i) && (i - middle).abs() <= 0.3 * middle && sized[k] && sized[k + 1])
        .collect();
    let accepted: Vec<f64> = intervals
        .iter()
        .zip(&good)
        .filter(|(_, ok)| **ok)
        .map(|(i, _)| *i)
        .collect();
    let quality = accepted.len() as f64 / intervals.len() as f64;
    if accepted.len() < 2 {
        return Heart {
            beats,
            quality,
            ..nothing
        };
    }
    let mean = accepted.iter().sum::<f64>() / accepted.len() as f64;
    let sdnn = (accepted.iter().map(|i| (i - mean).powi(2)).sum::<f64>() / accepted.len() as f64).sqrt();
    // Changes between neighbouring intervals, where both are trusted.
    let changes: Vec<f64> = (1..intervals.len())
        .filter(|k| good[*k] && good[k - 1])
        .map(|k| intervals[k] - intervals[k - 1])
        .collect();
    let rmssd = if changes.is_empty() {
        f64::NAN
    } else {
        (changes.iter().map(|c| c * c).sum::<f64>() / changes.len() as f64).sqrt()
    };
    Heart {
        beats,
        rate_bpm: 60.0 / mean,
        sdnn_ms: 1000.0 * sdnn,
        rmssd_ms: 1000.0 * rmssd,
        quality,
    }
}
