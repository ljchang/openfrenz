//! Filters and spectra against values computed with SciPy 1.x for the same signal.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, missing_docs)]

use frenz_model::dsp::{
    Asr, Design, Filter, Recent, Start, analytic, asr_calibrate, correlation, coupling, cross_spectrum, filtfilt,
    heart, morlet, periodogram, quality, scalogram, summary, welch,
};

const RATE: f64 = 125.0;
const AT: [usize; 5] = [0, 1, 7, 150, 399];

/// Two sines and a ramp: `sin(0.1 n) + 0.5 sin(1.3 n + 1) + 0.01 n`, 400 samples.
fn signal() -> Vec<f64> {
    (0..400)
        .map(|n| {
            let n = f64::from(n);
            (0.1 * n).sin() + 0.5 * (1.3 * n + 1.0).sin() + 0.01 * n
        })
        .collect()
}

fn close(got: &[f64], want: [f64; 5], at: [usize; 5], what: &str) {
    for (i, w) in at.iter().zip(want) {
        let g = got[*i];
        assert!(
            (g - w).abs() <= 1e-9 * w.abs().max(1e-3),
            "{what}[{i}]: {g} against {w}"
        );
    }
}

fn run(design: &Design, start: Start) -> Vec<f64> {
    Filter::new(design.clone(), start).process(&[signal()]).remove(0)
}

#[test]
fn butterworth_filters_match_scipy() {
    // (design, sosfilt, sosfilt from the settled state, sosfiltfilt)
    let cases = [
        (
            "band 0.5-45 order 2",
            Design::bandpass(2, 0.5, 45.0, RATE).unwrap(),
            [
                0.218732450860704,
                0.500972332784362,
                0.508278794249112,
                0.736663933742422,
                0.205620763192013,
            ],
            [
                -1.66533453693773e-16,
                0.0322069128374428,
                0.186548226171098,
                0.775362102699539,
                0.205671936565119,
            ],
            [
                0.465252996478147,
                0.488809657227763,
                0.368867167341864,
                1.10422467753342,
                0.0636357464800224,
            ],
        ),
        (
            "low 10 order 3",
            Design::lowpass(3, 10.0, RATE).unwrap(),
            [
                0.00428417143714578,
                0.0263521136959174,
                0.32278268381344,
                2.34596274362157,
                4.93138735232771,
            ],
            [
                0.420735492403948,
                0.421366308432909,
                0.318947493210574,
                2.34596274362156,
                4.93138735232771,
            ],
            [
                0.427514432884679,
                0.388359894769907,
                0.662896307275687,
                2.15094847547904,
                4.30991878826992,
            ],
        ),
        (
            "high 1 order 4",
            Design::highpass(4, 1.0, RATE).unwrap(),
            [
                0.393988307347692,
                0.400252753741783,
                0.049508835320554,
                -0.105877955268313,
                -0.907941755307401,
            ],
            [
                1.36184444951309e-17,
                0.0580121834862062,
                -0.0495713031969559,
                -0.0818674008523693,
                -0.907850448444546,
            ],
            [
                0.342710554133168,
                0.387802381988744,
                0.212523434724544,
                1.1101570203846,
                0.234577854850666,
            ],
        ),
        (
            "band 8-12 order 5",
            Design::bandpass(5, 8.0, 12.0, RATE).unwrap(),
            [
                3.1714582681999e-06,
                2.97546059568375e-05,
                -0.000689549188465869,
                0.000137440665355077,
                -1.50692504481009e-05,
            ],
            [
                0.0,
                4.66976343059967e-07,
                -0.000601579145456086,
                -0.000601477567686288,
                -1.57209471497625e-05,
            ],
            [
                -0.000197748982384564,
                -0.0639763291138486,
                0.0394500490937088,
                0.000712902265414442,
                0.00634249490485991,
            ],
        ),
    ];
    for (name, design, forward, settled, both_ways) in cases {
        close(&run(&design, Start::Zero), forward, AT, name);
        // The first settled sample is zero up to rounding where the filter blocks a constant.
        close(
            &run(&design, Start::Settled)[1..],
            [settled[1], settled[2], settled[3], settled[4], settled[4]],
            [0, 6, 149, 398, 398],
            name,
        );
        assert!((run(&design, Start::Settled)[0] - settled[0]).abs() < 1e-12, "{name}");
        close(&filtfilt(&design, &signal()), both_ways, AT, name);
    }
}

#[test]
fn notch_and_dc_block_match_scipy() {
    let notch = Design::notch(60.0, 12.0, RATE).unwrap();
    close(
        &run(&notch, Start::Zero),
        [
            0.373545696716531,
            0.511681053403146,
            0.435172506518261,
            2.60347308043798,
            4.33020590723602,
        ],
        AT,
        "notch",
    );
    let dc = Design::dc_block(0.99).unwrap();
    close(
        &run(&dc, Start::Zero),
        [
            0.420735492403948,
            0.478478667811149,
            0.371906685252229,
            1.79992885566257,
            1.23542714784449,
        ],
        AT,
        "dc",
    );
}

#[test]
fn filtering_in_pieces_equals_filtering_whole() {
    let design = Design::dc_block(0.99)
        .unwrap()
        .then(&Design::notch(60.0, 12.0, RATE).unwrap())
        .then(&Design::bandpass(2, 0.5, 45.0, RATE).unwrap());
    let x = signal();
    let two = vec![x.clone(), x.iter().map(|v| -2.0 * v).collect()];
    let whole = Filter::new(design.clone(), Start::Settled).process(&two);
    let mut filter = Filter::new(design.clone(), Start::Settled);
    let mut pieces: Vec<Vec<f64>> = vec![Vec::new(), Vec::new()];
    for from in (0..400).step_by(37) {
        let to = (from + 37).min(400);
        let out = filter.process(&[two[0][from..to].to_vec(), two[1][from..to].to_vec()]);
        pieces[0].extend(&out[0]);
        pieces[1].extend(&out[1]);
    }
    assert_eq!(pieces, whole);
    // After a reset the filter starts again from nothing.
    filter.reset();
    assert_eq!(filter.process(&two), whole);
}

#[test]
fn designs_refuse_impossible_requests() {
    assert!(Design::bandpass(2, 45.0, 0.5, RATE).is_err());
    assert!(Design::lowpass(2, 80.0, RATE).is_err());
    assert!(Design::highpass(0, 1.0, RATE).is_err());
    assert!(Design::notch(60.0, 0.0, RATE).is_err());
    assert!(Design::dc_block(1.0).is_err());
    assert_eq!(
        filtfilt(&Design::lowpass(3, 10.0, RATE).unwrap(), &[1.0, 2.0, 3.0]),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn spectra_match_scipy() {
    let x = signal();
    let p = periodogram(&x, RATE);
    assert_eq!(p.density.len(), 201);
    assert_eq!(p.resolution_hz, 0.3125);
    close(
        &p.density,
        [
            0.00297030477394935,
            0.974906435845878,
            0.0474634657599522,
            9.88204398507759e-09,
            9.88204398507759e-09,
        ],
        [0, 1, 2, 26, 26],
        "periodogram",
    );
    let w = welch(&x, RATE, 125);
    assert_eq!(w.density.len(), 63);
    assert_eq!(w.resolution_hz, 1.0);
    close(
        &w.density,
        [
            2.19441581633783e-05,
            0.216959276865522,
            0.378819915865643,
            0.0813320351357334,
            0.0813320351357334,
        ],
        [0, 1, 2, 26, 26],
        "welch",
    );
}

#[test]
fn band_power_adds_up_to_the_signal_variance() {
    // A 10 Hz sine of amplitude 3 has variance 4.5, all of it in the alpha band.
    let x: Vec<f64> = (0..1000)
        .map(|n| 3.0 * (2.0 * std::f64::consts::PI * 10.0 * f64::from(n) / RATE).sin())
        .collect();
    let spectrum = welch(&x, RATE, 250);
    assert!((spectrum.band_power(8.0, 12.0) - 4.5).abs() < 0.05);
    assert!(spectrum.band_power(20.0, 30.0) < 1e-6);
    let relative = spectrum.relative_band_power(&[(4.0, 8.0), (8.0, 12.0)], (1.0, 45.0));
    assert!(relative[0] < 0.01 && relative[1] > 0.99);
}

#[test]
fn envelope_and_phase_match_scipy() {
    let x = signal();
    let at = [0, 1, 7, 150, 398];
    let even = analytic(&x);
    close(
        &even.amplitude,
        [
            7.05107240995648,
            4.74174620447245,
            2.79454830903159,
            2.62587369426418,
            6.99960931688073,
        ],
        at,
        "amplitude",
    );
    close(
        &even.phase,
        [
            1.51109114458893,
            1.4688247050134,
            1.42655886777347,
            -0.0665963858101306,
            0.810406412018714,
        ],
        at,
        "phase",
    );
    let odd = analytic(&x[..399]);
    close(
        &odd.amplitude,
        [
            7.3066771345518,
            4.77864265220703,
            2.78983125930912,
            2.6256204689894,
            8.98610644271685,
        ],
        at,
        "amplitude, odd length",
    );
    close(
        &odd.phase,
        [
            1.51318212824279,
            1.46961474809846,
            1.4263132811041,
            -0.0651343149751148,
            1.00410159678072,
        ],
        at,
        "phase, odd length",
    );
    assert!(analytic(&[]).amplitude.is_empty());
}

#[test]
fn summaries_and_quality() {
    let s = summary(&signal());
    close(
        &[s.mean, s.sd, s.range, s.largest_step, s.mean],
        [
            2.0362597259956603,
            1.3867907116514102,
            6.396962214763625,
            0.7130725262719717,
            2.0362597259956603,
        ],
        [0, 1, 2, 3, 4],
        "summary",
    );
    let q = quality(&[1.0, 1.0, 1.0, 5.0, -8.0, f64::NAN, 2.0, 8.0], 8.0);
    assert_eq!((q.repeated, q.clipped, q.invalid), (0.25, 0.25, 0.125));
    assert_eq!(quality(&signal(), f64::INFINITY), frenz_model::dsp::Quality::default());
}

/// A second signal sharing the slow sine of [`signal`] at a different phase.
fn other() -> Vec<f64> {
    (0..400)
        .map(|n| {
            let n = f64::from(n);
            (0.1 * n + 0.5).sin() + 0.3 * (0.7 * n).sin() + 0.2 * (0.013 * n * n).sin()
        })
        .collect()
}

#[test]
fn coherence_matches_scipy_and_adds_up_over_steps() {
    let (x, y) = (signal(), other());
    let whole = cross_spectrum(&x, &y, RATE, 100);
    assert_eq!((whole.stretches, whole.resolution_hz), (7, 1.25));
    let coherence = whole.coherence();
    assert_eq!(coherence.len(), 51);
    for (k, want) in [
        (0, 0.772784762971896),
        (1, 0.912333094227104),
        (2, 0.992570765692503),
        (10, 0.0139922548019482),
        (26, 0.0964078021195517),
        (50, 0.0242588812771164),
    ] {
        assert!(
            (coherence[k] - want).abs() < 1e-9,
            "bin {k}: {} against {want}",
            coherence[k]
        );
    }
    // One stretch alone always looks perfectly coherent, which is why steps are summed.
    assert!(cross_spectrum(&x[..100], &y[..100], RATE, 100).coherence()[10] > 0.999);
    // Four steps of 100 samples, kept and summed, are the four whole stretches of the signal.
    let mut recent = Recent::new(4);
    for step in 0..4 {
        recent.push(cross_spectrum(
            &x[step * 100..(step + 1) * 100],
            &y[step * 100..(step + 1) * 100],
            RATE,
            100,
        ));
    }
    let total = recent.total().unwrap();
    assert!(recent.is_full() && total.stretches == 4);
    assert!(total.band_coherence(1.0, 4.0) > 0.9 && total.band_coherence(10.0, 20.0) < 0.5);
    // The shared sine is 0.5 rad later in the second signal.
    assert!(
        (total.band_phase_lag(1.0, 3.0) + 0.5).abs() < 0.1,
        "{}",
        total.band_phase_lag(1.0, 3.0)
    );
    recent.push(total.clone());
    assert_eq!(recent.iter().count(), 4);
}

#[test]
fn coupling_is_found_where_it_is_and_not_where_it_is_not() {
    // A 6 Hz rhythm and a 40 Hz rhythm whose size does or does not follow the 6 Hz phase.
    let t: Vec<f64> = (0..1000).map(|n| f64::from(n) / RATE).collect();
    let slow = |t: f64| (2.0 * std::f64::consts::PI * 6.0 * t).sin();
    let fast = |t: f64| 0.3 * (2.0 * std::f64::consts::PI * 40.0 * t).sin();
    let coupled: Vec<f64> = t.iter().map(|t| slow(*t) + (1.0 + 0.8 * slow(*t)) * fast(*t)).collect();
    let uncoupled: Vec<f64> = t.iter().map(|t| slow(*t) + fast(*t)).collect();
    let theta = Design::bandpass(2, 4.0, 8.0, RATE).unwrap();
    let gamma = Design::bandpass(2, 30.0, 50.0, RATE).unwrap();
    let measure = |x: &[f64]| {
        let phase = analytic(&filtfilt(&theta, x)).phase;
        let amplitude = analytic(&filtfilt(&gamma, x)).amplitude;
        let c = coupling(&phase[125..875], &amplitude[125..875], 18);
        let with_slow: Vec<f64> = phase[125..875].iter().map(|p| p.cos()).collect();
        (c, correlation(&amplitude[125..875], &with_slow))
    };
    // Values from the same computation in NumPy and SciPy.
    let (c, r) = measure(&coupled);
    assert!(
        (c.modulation_index - 0.0481399776253).abs() < 1e-7,
        "{}",
        c.modulation_index
    );
    assert!(
        (c.mean_vector_length - 0.322302472461).abs() < 1e-7,
        "{}",
        c.mean_vector_length
    );
    assert!((r - 0.999998015861).abs() < 1e-7);
    assert!(
        c.preferred_phase.abs() < 0.05,
        "largest at the peak of the slow rhythm: {}",
        c.preferred_phase
    );
    let (c, r) = measure(&uncoupled);
    assert!(c.modulation_index < 1e-8 && c.mean_vector_length < 1e-4 && r.abs() < 0.01);
    // Scaling the signal changes neither measure.
    let louder: Vec<f64> = coupled.iter().map(|v| 50.0 * v).collect();
    assert!((measure(&louder).0.mean_vector_length - 0.322302472461).abs() < 1e-7);
}

#[test]
fn a_wavelet_reads_the_size_of_a_rhythm_where_and_when_it_is() {
    // 10 Hz at amplitude 3 throughout, on an offset; 20 Hz at amplitude 2 in the second half only.
    let x: Vec<f64> = (0..1000)
        .map(|n| {
            let w = 2.0 * std::f64::consts::PI * f64::from(n) / RATE;
            500.0 + 3.0 * (10.0 * w).sin() + if n >= 500 { 2.0 * (20.0 * w).sin() } else { 0.0 }
        })
        .collect();
    let power = scalogram(&x, RATE, &[10.0, 20.0, 35.0], 6.0);
    assert!(
        (power[0][250] - 3.0).abs() < 0.01 && (power[0][750] - 3.0).abs() < 0.05,
        "{}",
        power[0][250]
    );
    // A 6-cycle wavelet at 20 Hz still sees about a hundredth of the 10 Hz rhythm.
    assert!(
        power[1][250] < 0.05 && (power[1][750] - 2.0).abs() < 0.05,
        "{} {}",
        power[1][250],
        power[1][750]
    );
    // 35 Hz is far from 10 Hz, and near enough to 20 Hz to see a little of it.
    assert!(power[2].iter().skip(100).take(300).all(|p| *p < 0.001));
    assert!(power[2].iter().skip(600).take(300).all(|p| *p < 0.1));
    // The phase advances with the rhythm: three samples of 10 Hz are 0.24 of a cycle.
    let w = morlet(&x, RATE, 10.0, 6.0);
    let step = (w.phase[253] - w.phase[250]).rem_euclid(2.0 * std::f64::consts::PI);
    assert!((step - 2.0 * std::f64::consts::PI * 10.0 * 3.0 / RATE).abs() < 1e-3);
    // Values from np.convolve with the same wavelet.
    close(
        &morlet(&signal(), RATE, 10.0, 6.0).amplitude,
        [
            0.2939915225859202,
            0.2930327744134829,
            0.24667226999198386,
            3.599997032794449e-05,
            0.3949438455173489,
        ],
        AT,
        "morlet",
    );
    assert!(morlet(&[], RATE, 10.0, 6.0).amplitude.is_empty());
}

/// A pulse signal at 25 Hz: one smooth pulse per beat, a drifting baseline and a faster ripple.
/// Beats come every 0.8 s on average, breathing in and out by 50 ms. Returns it with the beats.
fn pulse(seconds: f64) -> (Vec<f64>, Vec<f64>) {
    let mut beats = vec![0.5];
    while let Some(last) = beats.last().copied().filter(|t| *t < seconds) {
        beats.push(last + 0.8 + 0.05 * (2.0 * std::f64::consts::PI * 0.25 * last).sin());
    }
    let signal = (0..(seconds * 25.0) as usize)
        .map(|n| {
            let t = n as f64 / 25.0;
            let pulses: f64 = beats
                .iter()
                .map(|b| (-(t - b).powi(2) / (2.0 * 0.07 * 0.07)).exp())
                .sum();
            30000.0 + 400.0 * pulses + 300.0 * (0.1 * t).sin() + 15.0 * (2.0 * std::f64::consts::PI * 9.0 * t).sin()
        })
        .collect();
    (signal, beats)
}

#[test]
fn heartbeats_are_found_and_timed() {
    let (signal, truth) = pulse(60.0);
    let found = heart(&signal, 25.0);
    let inside: Vec<f64> = truth.iter().copied().filter(|t| *t > 1.0 && *t < 59.0).collect();
    // Every beat away from the ends is found, to within 10 ms.
    for beat in &inside {
        let nearest = found
            .beats
            .iter()
            .map(|b| (b - beat).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(nearest < 0.010, "beat at {beat}: nearest found {nearest} s away");
    }
    assert!(found.beats.len() <= inside.len() + 3);
    let intervals: Vec<f64> = inside.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    let sdnn = 1000.0 * (intervals.iter().map(|i| (i - mean).powi(2)).sum::<f64>() / intervals.len() as f64).sqrt();
    assert!(
        (found.rate_bpm - 60.0 / mean).abs() < 0.3,
        "{} against {}",
        found.rate_bpm,
        60.0 / mean
    );
    assert!((found.sdnn_ms - sdnn).abs() < 3.0, "{} against {sdnn}", found.sdnn_ms);
    assert!(found.rmssd_ms > 5.0 && found.rmssd_ms < 60.0, "{}", found.rmssd_ms);
    assert!(found.quality > 0.97, "{}", found.quality);
}

#[test]
fn a_spoiled_pulse_signal_says_so() {
    let (mut signal, _) = pulse(60.0);
    // Twenty seconds of movement: large swings unrelated to the pulse.
    for (n, v) in signal.iter_mut().enumerate().skip(500).take(500) {
        let n = n as f64;
        *v += 3000.0 * (0.9 * n).sin() * (0.37 * n).cos() + 2000.0 * (0.011 * n * n).sin();
    }
    let found = heart(&signal, 25.0);
    assert!(found.quality < 0.8, "{}", found.quality);
    // Nothing to find in a flat line or in too little signal.
    let flat = heart(&[30000.0; 500], 25.0);
    assert!(flat.beats.is_empty() && flat.rate_bpm.is_nan() && flat.quality == 0.0);
    assert!(heart(&[1.0, 2.0], 25.0).rate_bpm.is_nan());
}

/// Four mixed sources of smoothed noise, about 20 units in size; the same generator as in
/// `tests/fixture/make_asr_fixture.py`.
fn mixed_noise(seed: u64, samples: usize) -> Vec<Vec<f64>> {
    const MIX: [[f64; 4]; 4] = [
        [1.0, 0.6, 0.2, 0.1],
        [0.5, 1.0, 0.3, 0.2],
        [0.2, 0.4, 1.0, 0.5],
        [0.1, 0.2, 0.6, 1.0],
    ];
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let raw: Vec<[f64; 4]> = (0..samples + 2)
        .map(|_| std::array::from_fn(|_| (0..4).map(|_| next()).sum::<f64>() - 2.0))
        .collect();
    (0..4)
        .map(|c| {
            (0..samples)
                .map(|n| {
                    40.0 * (0..4)
                        .map(|s| MIX[c][s] * (raw[n][s] + raw[n + 1][s] + raw[n + 2][s]) / 3.0)
                        .sum::<f64>()
                })
                .collect()
        })
        .collect()
}

#[test]
fn artifact_reconstruction_matches_the_reference_and_spares_clean_data() {
    let expected: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixture/asr/expected.json")).unwrap(),
    )
    .unwrap();
    let list = |key: &str| -> Vec<Vec<f64>> { serde_json::from_value(expected[key].clone()).unwrap() };
    let calibration = asr_calibrate(&mixed_noise(1, 7500), RATE, 20.0).unwrap();
    for (got, want) in calibration.mixing.iter().flatten().zip(list("mixing").iter().flatten()) {
        assert!((got - want).abs() < 1e-8, "mixing: {got} against {want}");
    }
    let clean = mixed_noise(2, 1000);
    // The test signal is the fixture's, to rounding.
    for (c, want) in list("data")[2].iter().enumerate() {
        assert!((clean[c][420] - want).abs() < 1e-9);
    }
    // One second of a large slow wave on one channel.
    let mut dirty = clean.clone();
    for (k, v) in dirty[1].iter_mut().enumerate().skip(400).take(125) {
        *v += 600.0 * (2.0 * std::f64::consts::PI * 3.0 * (k - 400) as f64 / RATE).sin();
    }
    let mut asr = Asr::new(&calibration, 0.66, 0.5).unwrap();
    let cleaned = asr.process(&dirty, RATE).unwrap();
    let at: Vec<usize> = serde_json::from_value(expected["at"].clone()).unwrap();
    for (n, want) in at.iter().zip(list("cleaned")) {
        for c in 0..4 {
            assert!(
                (cleaned[c][*n] - want[c]).abs() < 1e-6,
                "sample {n}, channel {c}: {} against {}",
                cleaned[c][*n],
                want[c]
            );
        }
    }
    // The disturbance is mostly gone, and data away from it is left as it was.
    let error = |x: &[Vec<f64>]| (400..525).map(|n| (x[1][n] - clean[1][n]).powi(2)).sum::<f64>().sqrt();
    assert!(
        error(&cleaned) < 0.25 * error(&dirty),
        "{} against {}",
        error(&cleaned),
        error(&dirty)
    );
    for c in 0..4 {
        for n in (0..372).chain(700..1000) {
            assert!((cleaned[c][n] - dirty[c][n]).abs() < 1e-9);
        }
    }
    // In pieces it gives what it gives whole, when the pieces end where its windows do.
    asr.reset();
    let mut pieces = vec![Vec::new(); 4];
    for from in (0..1000).step_by(252) {
        let to = (from + 252).min(1000);
        let part: Vec<Vec<f64>> = dirty.iter().map(|c| c[from..to].to_vec()).collect();
        for (all, new) in pieces.iter_mut().zip(asr.process(&part, RATE).unwrap()) {
            all.extend(new);
        }
    }
    let worst = pieces
        .iter()
        .flatten()
        .zip(cleaned.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-9, "{worst}");
    assert!(asr_calibrate(&mixed_noise(1, 100), RATE, 20.0).is_err());
    assert!(asr.process(&dirty[..3], RATE).is_err());
}
