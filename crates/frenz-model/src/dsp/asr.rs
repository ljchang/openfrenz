//! Artifact subspace reconstruction: removing large, brief disturbances from multi-channel data.
//!
//! The method (Kothe & Jung; Mullen et al. 2015, IEEE Trans. Biomed. Eng.; evaluated by Chang et
//! al. 2020, same journal) learns from a clean stretch of data how large each direction of the
//! signal normally is. Later data is then looked at a short window at a time: directions that
//! are far larger than they were in the clean data are removed, and their content is rebuilt
//! from the directions that remain, using how the channels moved together in the clean data.
//!
//! This is a compact form of the method. It differs from the EEGLAB implementation in three
//! documented ways: thresholds come from the median and spread of the windowed sizes instead of
//! a fitted truncated Gaussian; each window is judged by its own covariance with no look-ahead,
//! so there is no delay; and no spectral weighting is applied, so filter the data first.

// Matrices are indexed inside the sizes they were just built with.
#![allow(clippy::indexing_slicing)]

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::Error;

/// What was learned from clean data: how channels mix, and how large each direction may get.
#[derive(Debug, Clone, PartialEq)]
pub struct AsrCalibration {
    /// Square root of the clean data's covariance, `[row][column]`, channels on both sides.
    pub mixing: Vec<Vec<f64>>,
    /// The limit of each direction of the clean data, as rows of `diag(limit) · directionsᵀ`.
    pub threshold: Vec<Vec<f64>>,
}

fn matrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let columns = rows.first().map_or(0, Vec::len);
    DMatrix::from_fn(rows.len(), columns, |r, c| {
        rows.get(r).and_then(|row| row.get(c)).copied().unwrap_or(0.0)
    })
}

fn rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    m.row_iter().map(|r| r.iter().copied().collect()).collect()
}

/// Eigenvalues in rising order with their vectors as columns.
fn eigen(m: DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let decomposition = SymmetricEigen::new(m);
    let mut order: Vec<usize> = (0..decomposition.eigenvalues.len()).collect();
    order.sort_by(|a, b| decomposition.eigenvalues[*a].total_cmp(&decomposition.eigenvalues[*b]));
    let values = DVector::from_iterator(order.len(), order.iter().map(|i| decomposition.eigenvalues[*i]));
    let vectors = DMatrix::from_fn(order.len(), order.len(), |r, c| {
        decomposition.eigenvectors[(r, order[c])]
    });
    (values, vectors)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    match n {
        0 => f64::NAN,
        _ if n % 2 == 1 => values[n / 2],
        _ => 0.5 * (values[n / 2 - 1] + values[n / 2]),
    }
}

/// The point with the least summed distance to `points` (each a flattened matrix): a centre that
/// a minority of outlying blocks cannot move far.
fn geometric_median(points: &[DVector<f64>]) -> DVector<f64> {
    let mut centre = points.iter().fold(DVector::zeros(points[0].len()), |sum, p| sum + p) / points.len() as f64;
    for _ in 0..1000 {
        let (mut weighted, mut weights) = (DVector::zeros(centre.len()), 0.0);
        for p in points {
            let w = 1.0 / (p - &centre).norm().max(1e-12);
            weighted += p * w;
            weights += w;
        }
        let next = weighted / weights;
        let moved = (&next - &centre).norm();
        centre = next;
        if moved <= 1e-12 * centre.norm() {
            break;
        }
    }
    centre
}

/// Learns the calibration from `reference`: clean data, `[channel][sample]`, at least a minute
/// of it if possible, filtered as the data to be cleaned will be (in particular without drift
/// or offset).
///
/// `cutoff` says how many times its usual spread a direction may exceed its usual size before
/// it is removed. 20 to 30 is the recommended range (Chang et al. 2020); lower values remove
/// more and begin to remove brain signal.
pub fn asr_calibrate(reference: &[Vec<f64>], rate_hz: f64, cutoff: f64) -> Result<AsrCalibration, Error> {
    let channels = reference.len();
    let samples = reference.iter().map(Vec::len).min().unwrap_or(0);
    let window = (0.5 * rate_hz).round() as usize;
    if channels == 0 || window == 0 || samples < 10 * window {
        return Err(Error::Spec(
            "artifact reconstruction needs at least five seconds of clean data".into(),
        ));
    }
    let x = DMatrix::from_fn(channels, samples, |c, n| reference[c][n]);
    // The covariance of the clean data, as the centre of the covariances of short blocks, so
    // that the few disturbed blocks any recording has do not count.
    const BLOCK: usize = 10;
    let blocks: Vec<DVector<f64>> = (0..samples / BLOCK)
        .map(|b| {
            let part = x.columns(b * BLOCK, BLOCK);
            let covariance = part * part.transpose() / BLOCK as f64;
            DVector::from_column_slice(covariance.as_slice())
        })
        .collect();
    let covariance = DMatrix::from_column_slice(channels, channels, geometric_median(&blocks).as_slice());
    let (values, vectors) = eigen(covariance);
    let roots = DMatrix::from_diagonal(&values.map(|v| v.max(0.0).sqrt()));
    let mixing = &vectors * roots * vectors.transpose();
    // Directions of the clean data and how large each is, half a second at a time.
    let (_, directions) = eigen(mixing.clone());
    let activity = directions.transpose() * &x;
    let hop = ((window as f64) * 0.34).round().max(1.0) as usize;
    let mut limits = Vec::with_capacity(channels);
    for c in 0..channels {
        let mut sizes: Vec<f64> = (0..=(samples - window) / hop)
            .map(|w| {
                let part = activity.row(c).columns(w * hop, window).into_owned();
                (part.iter().map(|v| v * v).sum::<f64>() / window as f64).sqrt()
            })
            .collect();
        let middle = median(&mut sizes);
        let mut off: Vec<f64> = sizes.iter().map(|s| (s - middle).abs()).collect();
        limits.push(middle + cutoff * 1.4826 * median(&mut off));
    }
    let threshold = DMatrix::from_diagonal(&DVector::from_vec(limits)) * directions.transpose();
    Ok(AsrCalibration {
        mixing: rows(&mixing),
        threshold: rows(&threshold),
    })
}

/// Artifact subspace reconstruction running over data in pieces.
#[derive(Debug, Clone)]
pub struct Asr {
    mixing: DMatrix<f64>,
    threshold: DMatrix<f64>,
    /// Most directions removed at once.
    most: usize,
    window_s: f64,
    /// The reconstruction applied at the end of the previous window; none before the first.
    previous: Option<DMatrix<f64>>,
}

impl Asr {
    /// A cleaner from a calibration. `max_removed` is the largest share of directions that may
    /// be removed at once (0.66 is usual); `window_s` the length of the windows the data is
    /// judged in (0.5 s is usual).
    pub fn new(calibration: &AsrCalibration, max_removed: f64, window_s: f64) -> Result<Self, Error> {
        let (mixing, threshold) = (matrix(&calibration.mixing), matrix(&calibration.threshold));
        let channels = mixing.nrows();
        if channels == 0 || !mixing.is_square() || threshold.shape() != mixing.shape() {
            return Err(Error::Spec(
                "the artifact calibration must be two square matrices of one size".into(),
            ));
        }
        if !(0.0..=1.0).contains(&max_removed) || window_s.is_nan() || window_s <= 0.0 {
            return Err(Error::Spec(
                "max_removed must lie between 0 and 1 and window_s be positive".into(),
            ));
        }
        Ok(Self {
            mixing,
            threshold,
            most: (max_removed * channels as f64).floor() as usize,
            window_s,
            previous: None,
        })
    }

    /// Forgets the previous window.
    pub fn reset(&mut self) {
        self.previous = None;
    }

    /// The reconstruction for one window of data: the identity when nothing is too large.
    fn reconstruction(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let channels = x.nrows();
        let (values, vectors) = eigen(x * x.transpose() / x.ncols() as f64);
        let limits = &self.threshold * &vectors;
        // Rising order: the smallest directions are always kept, so that never more than
        // `most` are removed.
        let keep: Vec<bool> = (0..channels)
            .map(|i| i < channels - self.most || values[i] < limits.column(i).iter().map(|v| v * v).sum::<f64>())
            .collect();
        if keep.iter().all(|k| *k) {
            return DMatrix::identity(channels, channels);
        }
        let mut kept = vectors.transpose() * &self.mixing;
        for (i, keep) in keep.iter().enumerate() {
            if !keep {
                kept.row_mut(i).fill(0.0);
            }
        }
        let svd = kept.svd(true, true);
        let largest = svd.singular_values.iter().copied().fold(0.0, f64::max);
        match svd.pseudo_inverse(1e-10 * largest) {
            Ok(inverse) => &self.mixing * inverse * vectors.transpose(),
            Err(_) => DMatrix::identity(channels, channels),
        }
    }

    /// Cleans the next samples (`x[channel][sample]`, at `rate_hz`), continuing from the
    /// previous call. The channels must be those of the calibration, in the same order.
    pub fn process(&mut self, x: &[Vec<f64>], rate_hz: f64) -> Result<Vec<Vec<f64>>, Error> {
        let channels = self.mixing.nrows();
        if x.len() != channels {
            return Err(Error::Spec(format!(
                "the artifact calibration is for {channels} channels and was given {}",
                x.len()
            )));
        }
        let samples = x.iter().map(Vec::len).min().unwrap_or(0);
        let window = ((self.window_s * rate_hz).round() as usize).max(1);
        let data = DMatrix::from_fn(channels, samples, |c, n| {
            x.get(c).and_then(|r| r.get(n)).copied().unwrap_or(0.0)
        });
        let mut out = DMatrix::zeros(channels, samples);
        let mut from = 0;
        while from < samples {
            // A last piece too short to judge on its own joins the window before it.
            let to = if samples - from < 2 * window {
                samples
            } else {
                from + window
            };
            let part = data.columns(from, to - from).into_owned();
            let next = self.reconstruction(&part);
            let before = self.previous.take().unwrap_or_else(|| next.clone());
            let length = part.ncols();
            for k in 0..length {
                // Ease from the previous reconstruction into this one across the window.
                let blend = 0.5 - 0.5 * (std::f64::consts::PI * (k + 1) as f64 / length as f64).cos();
                let sample = part.column(k);
                out.set_column(from + k, &(&next * sample * blend + &before * sample * (1.0 - blend)));
            }
            self.previous = Some(next);
            from = to;
        }
        Ok(rows(&out))
    }
}
