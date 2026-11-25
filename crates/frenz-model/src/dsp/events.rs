//! Moments in a signal: peaks that stand out from their surroundings, and abrupt changes of
//! level.
//!
//! Both detectors take one sample at a time and carry their state, so a signal fed in pieces of
//! any size gives the same moments as the signal fed whole. Each reports a moment a fixed number
//! of samples after it happened, because it has to see what follows before it can judge.

use std::collections::VecDeque;

/// The ordinary size of a signal that now and then does something much larger: a running median
/// of the absolute value, scaled to equal the standard deviation of Gaussian noise.
///
/// A mean would be pulled up by the very events one wants to find. The median is tracked by
/// nudging the estimate up or down by a fixed fraction with each sample, which settles where
/// half the samples lie on each side.
#[derive(Debug, Clone)]
pub struct RunningScale {
    /// Fraction the estimate moves by with each sample, as a logarithm.
    nudge: f64,
    warmup: usize,
    first: Vec<f64>,
    median: f64,
}

/// A median of absolute values times this is the standard deviation of Gaussian noise.
const MEDIAN_TO_SD: f64 = 1.4826;

impl RunningScale {
    /// A scale that follows slow changes over about `half_life` samples, and that starts from
    /// the median of the first `warmup` samples.
    pub fn new(half_life: f64, warmup: usize) -> Self {
        Self {
            nudge: 2.0 * std::f64::consts::LN_2 / half_life.max(1.0),
            warmup: warmup.max(1),
            first: Vec::new(),
            median: 0.0,
        }
    }

    /// Forgets everything.
    pub fn reset(&mut self) {
        self.first.clear();
        self.median = 0.0;
    }

    /// Whether the first samples have been gathered.
    pub fn ready(&self) -> bool {
        self.first.len() >= self.warmup
    }

    /// Takes one sample and returns the scale.
    pub fn push(&mut self, x: f64) -> f64 {
        let size = x.abs();
        if !size.is_finite() {
            return self.value();
        }
        if self.ready() {
            // A tiny floor keeps an estimate that reached zero able to grow again.
            let median = self.median.max(f64::MIN_POSITIVE);
            self.median = median
                * if size > median {
                    self.nudge.exp()
                } else {
                    (-self.nudge).exp()
                };
        } else {
            self.first.push(size);
            if self.ready() {
                self.first.sort_by(f64::total_cmp);
                let n = self.first.len();
                let middle = |k: usize| self.first.get(k).copied().unwrap_or_default();
                self.median = if n.is_multiple_of(2) {
                    (middle(n / 2 - 1) + middle(n / 2)) / 2.0
                } else {
                    middle(n / 2)
                };
            }
        }
        self.value()
    }

    /// The scale after the last sample; zero until it is ready.
    pub fn value(&self) -> f64 {
        if self.ready() { MEDIAN_TO_SD * self.median } else { 0.0 }
    }
}

/// A peak found by [`Peaks`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    /// Which sample it was, counting from zero at the start or the last reset.
    pub index: u64,
    /// How far it stands above its surroundings, in units of the signal's ordinary size.
    pub size: f64,
}

/// Settings of [`Peaks`], in samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakSettings {
    /// A peak must stand this many times the signal's ordinary size above its surroundings.
    pub threshold: f64,
    /// How far before the peak to look for the lowest point on that side.
    pub back: usize,
    /// How far after the peak to look for the lowest point on that side.
    pub ahead: usize,
    /// Of several peaks closer together than this, only the highest counts.
    pub min_gap: usize,
    /// How slowly the ordinary size follows the signal.
    pub scale_half_life: f64,
    /// Samples gathered before anything is detected.
    pub warmup: usize,
}

/// Finds peaks that stand out from their surroundings: local maxima whose height above the
/// higher of the two lowest points nearby (one before, one after) is at least `threshold` times
/// the signal's ordinary size.
#[derive(Debug, Clone)]
pub struct Peaks {
    settings: PeakSettings,
    scale: RunningScale,
    /// The last `back + ahead + 1` samples.
    recent: VecDeque<f64>,
    seen: u64,
    /// A peak waiting to learn whether a higher one follows within `min_gap`: index, height, size.
    held: Option<(u64, f64, f64)>,
}

impl Peaks {
    /// A detector with these settings.
    pub fn new(settings: PeakSettings) -> Self {
        let settings = PeakSettings {
            back: settings.back.max(1),
            ahead: settings.ahead.max(1),
            ..settings
        };
        Self {
            scale: RunningScale::new(settings.scale_half_life, settings.warmup),
            recent: VecDeque::with_capacity(settings.back + settings.ahead + 1),
            seen: 0,
            held: None,
            settings,
        }
    }

    /// Forgets everything; sample numbers start again from zero.
    pub fn reset(&mut self) {
        self.scale.reset();
        self.recent.clear();
        self.seen = 0;
        self.held = None;
    }

    /// Samples taken since the start or the last reset.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Takes one sample. Returns a peak once it is certain, which is `ahead` to
    /// `ahead + min_gap` samples after it happened.
    pub fn push(&mut self, x: f64) -> Option<Peak> {
        let PeakSettings {
            threshold,
            back,
            ahead,
            min_gap,
            ..
        } = self.settings;
        let scale = self.scale.push(x);
        if self.recent.len() == back + ahead + 1 {
            self.recent.pop_front();
        }
        self.recent.push_back(x);
        self.seen += 1;
        if self.recent.len() < back + ahead + 1 {
            return None;
        }
        // The sample being judged is `ahead` behind the newest.
        let index = self.seen - 1 - ahead as u64;
        let mut found = None;
        if let Some((held, _, size)) = self.held
            && index - held >= min_gap as u64
        {
            found = Some(Peak { index: held, size });
            self.held = None;
        }
        let at = |k: usize| self.recent.get(k).copied().unwrap_or(f64::NAN);
        let here = at(back);
        if self.scale.ready() && here >= at(back - 1) && here > at(back + 1) {
            let lowest = |from: usize, to: usize| (from..to).map(at).fold(f64::INFINITY, f64::min);
            let stands = here - lowest(0, back).max(lowest(back + 1, back + ahead + 1));
            if stands >= threshold * scale && self.held.is_none_or(|(_, height, _)| here > height) {
                self.held = Some((index, here, if scale > 0.0 { stands / scale } else { f64::INFINITY }));
            }
        }
        found
    }
}

/// Settings of [`Jumps`], in samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JumpSettings {
    /// A change between two samples counts when it is this many times the ordinary change.
    pub threshold: f64,
    /// Of several such changes closer together than this, only the largest counts.
    pub min_gap: usize,
    /// The level before and after a jump is the mean from `near` to `far` samples away from it.
    pub near: usize,
    /// See `near`.
    pub far: usize,
    /// Samples after which half of a jump has faded from the position.
    pub leak_half_life: f64,
    /// How slowly the ordinary change follows the signal.
    pub scale_half_life: f64,
    /// Samples gathered before anything is detected.
    pub warmup: usize,
}

/// Follows a level that moves in abrupt jumps while the sensor's own baseline wanders: each
/// jump's height is added to a position that slowly fades back to zero.
///
/// Slow wander never reaches the position, because only the jumps are added; an eye that turns
/// and stays turned is still seen, which a high-pass filter would lose.
#[derive(Debug, Clone)]
pub struct Jumps {
    settings: JumpSettings,
    scale: RunningScale,
    /// Samples kept to measure the level on each side of a jump.
    recent: VecDeque<f64>,
    keep: usize,
    seen: u64,
    /// The largest change seen lately, waiting out `min_gap`: index and size.
    candidate: Option<(u64, f64)>,
    /// Jumps waiting for the samples that follow them.
    pending: VecDeque<u64>,
    decay: f64,
    position: f64,
}

impl Jumps {
    /// A follower with these settings.
    pub fn new(settings: JumpSettings) -> Self {
        let far = settings.far.max(2);
        let settings = JumpSettings {
            far,
            near: settings.near.min(far - 1),
            ..settings
        };
        let keep = far + far.max(settings.min_gap) + 2;
        Self {
            scale: RunningScale::new(settings.scale_half_life, settings.warmup),
            recent: VecDeque::with_capacity(keep),
            keep,
            seen: 0,
            candidate: None,
            pending: VecDeque::new(),
            decay: 0.5f64.powf(1.0 / settings.leak_half_life.max(1.0)),
            position: 0.0,
            settings,
        }
    }

    /// Forgets everything; the position returns to zero.
    pub fn reset(&mut self) {
        self.scale.reset();
        self.recent.clear();
        self.seen = 0;
        self.candidate = None;
        self.pending.clear();
        self.position = 0.0;
    }

    /// The position after the last sample.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Mean of the samples numbered `from..to`, if they are all still kept.
    fn mean(&self, from: u64, to: u64) -> Option<f64> {
        let oldest = self.seen - self.recent.len() as u64;
        if from < oldest || to > self.seen || to <= from {
            return None;
        }
        let values = (from..to).filter_map(|k| self.recent.get((k - oldest) as usize));
        Some(values.sum::<f64>() / (to - from) as f64)
    }

    /// Takes one sample. Returns whether a jump was added to the position, which happens `far`
    /// samples after the jump.
    pub fn push(&mut self, x: f64) -> bool {
        let JumpSettings {
            threshold,
            min_gap,
            near,
            far,
            ..
        } = self.settings;
        let before = self.recent.back().copied();
        if self.recent.len() == self.keep {
            self.recent.pop_front();
        }
        self.recent.push_back(x);
        let index = self.seen;
        self.seen += 1;
        self.position *= self.decay;
        let Some(before) = before else { return false };
        let change = (x - before).abs();
        let scale = self.scale.push(x - before);
        if let Some((at, _)) = self.candidate
            && index - at >= min_gap as u64
        {
            self.pending.push_back(at);
            self.candidate = None;
        }
        if self.scale.ready() && change >= threshold * scale && self.candidate.is_none_or(|(_, size)| change > size) {
            self.candidate = Some((index, change));
        }
        let mut jumped = false;
        while let Some(at) = self.pending.front().copied() {
            if index - at < far as u64 {
                break;
            }
            self.pending.pop_front();
            if at < far as u64 {
                continue;
            }
            let after = self.mean(at + near as u64, at + far as u64);
            let earlier = self.mean(at - far as u64, at - near as u64);
            if let (Some(after), Some(earlier)) = (after, earlier) {
                self.position += after - earlier;
                jumped = true;
            }
        }
        jumped
    }
}
