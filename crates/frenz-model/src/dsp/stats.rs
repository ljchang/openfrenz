//! Summary numbers of a stretch of signal, and checks of its quality.

/// Level and spread of a stretch of signal.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Summary {
    /// Mean.
    pub mean: f64,
    /// Standard deviation (dividing by the number of samples).
    pub sd: f64,
    /// Largest value minus smallest.
    pub range: f64,
    /// Largest change between two consecutive samples.
    pub largest_step: f64,
}

/// Summary of `x`. All zeros for an empty signal.
pub fn summary(x: &[f64]) -> Summary {
    if x.is_empty() {
        return Summary::default();
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let (low, high) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| (l.min(*v), h.max(*v)));
    Summary {
        mean,
        sd: (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt(),
        range: high - low,
        largest_step: x
            .windows(2)
            .filter_map(|w| Some((w.get(1)? - w.first()?).abs()))
            .fold(0.0, f64::max),
    }
}

/// How trustworthy a stretch of signal is. Each number is a fraction of the samples, so the
/// same limits work for any length.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Quality {
    /// Samples equal to the one before: a frozen or disconnected sensor repeats itself.
    pub repeated: f64,
    /// Samples at or beyond the limit: the sensor ran out of range.
    pub clipped: f64,
    /// Samples that are not numbers.
    pub invalid: f64,
}

/// Quality of `x`, counting as clipped every sample whose size reaches `limit` (the largest
/// value the sensor can report; infinity if unknown).
pub fn quality(x: &[f64], limit: f64) -> Quality {
    if x.is_empty() {
        return Quality::default();
    }
    let n = x.len() as f64;
    let count = |hit: usize| hit as f64 / n;
    Quality {
        repeated: count(x.windows(2).filter(|w| w.first() == w.get(1)).count()),
        clipped: count(x.iter().filter(|v| v.abs() >= limit).count()),
        invalid: count(x.iter().filter(|v| !v.is_finite()).count()),
    }
}
