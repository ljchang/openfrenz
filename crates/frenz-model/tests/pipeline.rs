//! A pipeline read from files, against the same computation in NumPy, SciPy and PyTorch
//! (`tests/fixture/make_pipeline_fixture.py`).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic, missing_docs)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use frenz_model::dsp::{self, Asr, Design};
use frenz_model::{Block, Error, Input, Model, Pipeline, Registry, Step, Stepper, Take};
use serde_json::{Value, json};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixture/pipeline")
}

/// The fixture's signals: the same formulas as in the Python script.
fn step(index: usize) -> Step {
    let eeg: Vec<Vec<f64>> = (0..4)
        .map(|c| {
            let c = f64::from(c);
            (0..250)
                .map(|n| {
                    let w = 2.0 * PI * (index * 250 + n) as f64 / 125.0;
                    2000.0 * (10.0 * w + c).sin()
                        + 800.0 * (6.0 * w).sin() * (1.0 + 0.5 * (0.3 * w + c).sin())
                        + 300.0 * (40.0 * w * (1.0 + 0.01 * c)).sin() * (1.0 + 0.6 * (6.0 * w).sin())
                        + 5000.0
                        + 100.0 * c
                })
                .collect()
        })
        .collect();
    let accel: Vec<Vec<f64>> = (0..3)
        .map(|a| {
            (0..100)
                .map(|n| {
                    let w = 2.0 * PI * (index * 100 + n) as f64 / 50.0;
                    (if a == 0 { 16000.0 } else { 0.0 })
                        + 50.0 * (1.5 * w + f64::from(a)).sin()
                        + 20.0 * (7.0 * w).sin()
                })
                .collect()
        })
        .collect();
    let block = |values: Vec<Vec<f64>>| Block {
        t: vec![0.0; values[0].len()],
        values,
    };
    Step::new(
        0.0,
        0.0,
        [("eeg".to_owned(), block(eeg)), ("accel".to_owned(), block(accel))],
    )
}

fn expected(key: &str) -> Vec<Option<Vec<f64>>> {
    let all: Value = serde_json::from_str(&std::fs::read_to_string(fixture().join("expected.json")).unwrap()).unwrap();
    serde_json::from_value(all[key].clone()).unwrap()
}

fn compare(model: &mut Pipeline, want: &[Option<Vec<f64>>], tolerance: f64) {
    for (index, want) in want.iter().enumerate() {
        let got = model.step(&step(index)).unwrap();
        match (got, want) {
            (None, None) => {}
            (Some(got), Some(want)) => {
                assert_eq!(got.len(), want.len());
                for (k, (g, w)) in got.iter().zip(want).enumerate() {
                    let name = &model.spec().outputs[k];
                    assert!(
                        (g - w).abs() <= tolerance * w.abs().max(1.0),
                        "step {index}, {name}: {g} against {w}"
                    );
                }
            }
            (got, want) => panic!("step {index}: {got:?} against {want:?}"),
        }
    }
}

/// The fixture's config with `change` applied, in a folder of its own beside the stored numbers.
fn variant(change: impl FnOnce(&mut Value)) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut config: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    change(&mut config);
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    std::fs::copy(
        fixture().join("model.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .unwrap();
    dir
}

#[test]
fn a_pipeline_gives_what_the_reference_computes() {
    let mut model = Pipeline::from_dir(&fixture()).unwrap();
    assert_eq!(model.spec().outputs, ["a", "b"]);
    assert_eq!(model.spec().inputs.len(), 2);
    assert!((model.spec().max_step_seconds - 2.2).abs() < 1e-12);
    // The network runs in double precision on single-precision weights; PyTorch in single.
    compare(&mut model, &expected("outputs"), 1e-5);
    // After a reset it starts over exactly: same warm-up, same numbers.
    model.reset();
    compare(&mut model, &expected("outputs"), 1e-5);
}

#[test]
fn every_feature_matches_the_reference() {
    let dir = variant(|config| config["outputs"] = json!(["x"]));
    let mut model = Pipeline::from_dir(dir.path()).unwrap();
    let names = &model.spec().outputs;
    assert_eq!(names.len(), 35);
    assert_eq!(names[0], "forehead_right:power:4-8");
    assert_eq!(names[12], "forehead_right:envelope:20-30:mean");
    assert_eq!(names[20], "forehead_left~forehead_right:coherence:8-12");
    assert_eq!(names[21], "forehead_right:coupling:4-8/30-45:modulation_index");
    assert_eq!(names[34], "z:mean");
    compare(&mut model, &expected("features"), 1e-7);
}

#[test]
fn a_registry_loads_pipelines_by_their_files() {
    let model = Registry::new().with_pipelines().load_dir(&fixture()).unwrap();
    assert_eq!(model.spec().name, "fixture");
}

#[test]
fn mistakes_in_a_config_are_reported_when_it_is_loaded() {
    let message = |change: fn(&mut Value)| match Pipeline::from_dir(variant(change).path()) {
        Err(Error::Files(message)) => message,
        Err(other) => panic!("{other}"),
        Ok(_) => panic!("loaded"),
    };
    // A network fed the wrong number of features.
    let wrong_size = message(|c| c["nodes"][6]["in"] = json!(["power", "move"]));
    assert!(
        wrong_size.contains("node z") && wrong_size.contains("35 features and given 18"),
        "{wrong_size}"
    );
    assert!(
        message(|c| c["nodes"][1]["in"] = json!("nowhere")).contains("neither an input stream nor an earlier node")
    );
    assert!(message(|c| c["nodes"][1]["op"] = json!("magic")).contains("node power: unknown op"));
    assert!(message(|c| c["nodes"][0]["steps"][2]["high"] = json!(80)).contains("node clean"));
    assert!(message(|c| c["nodes"][3]["pairs"] = json!([["forehead_left", "nose"]])).contains("no channel \"nose\""));
    assert!(
        message(|c| c["nodes"][9]["layers"][2]["weights"] = json!("missing")).contains("no array \"missing.weight\"")
    );
    assert!(message(|c| c["step"]["stream"] = json!("battery")).contains("steady rate"));
    assert!(message(|c| c["outputs"] = json!(["clean"])).contains("not features"));
    // Sizes a file asks for are bounded before anything is built.
    assert!(message(|c| c["step"]["samples"] = json!(4_000_000_000u64)).contains("between 1 and 100000"));
    assert!(message(|c| c["nodes"][8]["steps"] = json!(1_000_000_000u64)).contains("node recent"));
    assert!(message(|c| c["nodes"][0]["steps"][2]["order"] = json!(5000)).contains("between 1 and 16"));
    assert!(message(|c| c["nodes"][3]["segment"] = json!(0)).contains("node coh"));
    assert!(
        message(|c| {
            c["nodes"][1] =
                json!({"name": "power", "op": "wavelet", "in": "clean", "frequencies": [1e-9], "cycles": 6});
        })
        .contains("not a frequency")
    );
}

#[test]
fn a_running_normalizer_warms_up_and_then_follows_the_data() {
    let dir = variant(|config| {
        config["nodes"] = json!([
            {"name": "move", "op": "summary", "in": "accel", "stats": ["mean"]},
            {"name": "z", "op": "normalize", "in": "move", "half_life_steps": 20, "warmup_steps": 3},
        ]);
        config["outputs"] = json!(["z"]);
    });
    let mut model = Pipeline::from_dir(dir.path()).unwrap();
    let outputs: Vec<_> = (0..8).map(|i| model.step(&step(i)).unwrap()).collect();
    assert!(outputs[..3].iter().all(Option::is_none) && outputs[3..].iter().all(Option::is_some));
    assert!(
        outputs[7]
            .as_ref()
            .unwrap()
            .iter()
            .all(|z| z.is_finite() && z.abs() < 5.0)
    );
}

/// A folder holding `config` and, if given, stored arrays `(name, shape, values)`.
fn model_dir(config: &Value, arrays: &[(&str, Vec<usize>, Vec<f64>)]) -> tempfile::TempDir {
    use safetensors::tensor::{Dtype, TensorView};
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    if !arrays.is_empty() {
        let raw: Vec<Vec<u8>> = arrays
            .iter()
            .map(|(_, _, v)| v.iter().flat_map(|x| x.to_le_bytes()).collect())
            .collect();
        let views = arrays.iter().zip(&raw).map(|((name, shape, _), bytes)| {
            (
                (*name).to_owned(),
                TensorView::new(Dtype::F64, shape.clone(), bytes).unwrap(),
            )
        });
        std::fs::write(
            dir.path().join("model.safetensors"),
            safetensors::serialize(views, None).unwrap(),
        )
        .unwrap();
    }
    dir
}

fn eeg_of(index: usize) -> Vec<Vec<f64>> {
    step(index).get("eeg").unwrap().values.clone()
}

fn eeg_step(values: Vec<Vec<f64>>) -> Step {
    let block = Block {
        t: vec![0.0; 250],
        values,
    };
    Step::new(0.0, 0.0, [("eeg".to_owned(), block)])
}

#[test]
fn wavelet_heart_and_artifact_nodes_give_what_the_functions_give() {
    // Wavelet: the mean amplitude at each frequency, away from the ends of the step.
    let config = json!({"openfrenz_model": "pipeline", "step": {"stream": "eeg", "samples": 250}, "nodes": [
        {"name": "w", "op": "wavelet", "in": "eeg", "frequencies": [6, 10], "cycles": 5, "trim_s": 0.4}], "outputs": ["w"]});
    let mut model = Pipeline::from_dir(model_dir(&config, &[]).path()).unwrap();
    assert_eq!(
        model.spec().outputs[..2],
        ["forehead_right:wavelet:6", "forehead_right:wavelet:10"]
    );
    let got = model.step(&step(0)).unwrap().unwrap();
    let x = &eeg_of(0)[0];
    for (k, hz) in [6.0, 10.0].into_iter().enumerate() {
        let amplitude = dsp::morlet(x, 125.0, hz, 5.0).amplitude;
        let want = amplitude[50..200].iter().sum::<f64>() / 150.0;
        assert!(
            (got[k] - want).abs() < 1e-9 * want,
            "{hz} Hz: {} against {want}",
            got[k]
        );
    }
    assert!(
        (got[1] - 2000.0).abs() < 100.0,
        "the test signal's 10 Hz rhythm has amplitude 2000: {}",
        got[1]
    );

    // Artifact reconstruction: the stored calibration applied step after step.
    let band = || dsp::Filter::new(Design::bandpass(2, 1.0, 45.0, 125.0).unwrap(), dsp::Start::Settled);
    let mut quiet = vec![Vec::new(); 4];
    let mut filter = band();
    for index in 0..20 {
        for (channel, new) in quiet.iter_mut().zip(filter.process(&eeg_of(index))) {
            channel.extend(new);
        }
    }
    let calibration = dsp::asr_calibrate(&quiet, 125.0, 20.0).unwrap();
    let flat = |m: &[Vec<f64>]| m.iter().flatten().copied().collect::<Vec<f64>>();
    let config = json!({"openfrenz_model": "pipeline", "step": {"stream": "eeg", "samples": 250}, "nodes": [
        {"name": "clean", "op": "filter", "in": "eeg", "start": "settled", "steps": [{"type": "bandpass", "low": 1, "high": 45}]},
        {"name": "fixed", "op": "asr", "in": "clean", "weights": "calibration"},
        {"name": "size", "op": "summary", "in": "fixed", "stats": ["sd"]}], "outputs": ["size"]});
    let dir = model_dir(
        &config,
        &[
            ("calibration.mixing", vec![4, 4], flat(&calibration.mixing)),
            ("calibration.threshold", vec![4, 4], flat(&calibration.threshold)),
        ],
    );
    let mut model = Pipeline::from_dir(dir.path()).unwrap();
    let (mut filter, mut asr) = (band(), Asr::new(&calibration, 0.66, 0.5).unwrap());
    let mut reduced = false;
    for index in 0..4 {
        let mut eeg = eeg_of(index);
        if index == 2 {
            eeg[1].iter_mut().skip(60).take(100).for_each(|v| *v += 60000.0); // a jolt on one channel
        }
        let got = model.step(&eeg_step(eeg.clone())).unwrap().unwrap();
        let filtered = filter.process(&eeg);
        let cleaned = asr.process(&filtered, 125.0).unwrap();
        for (g, x) in got.iter().zip(&cleaned) {
            let want = dsp::summary(x).sd;
            assert!((g - want).abs() < 1e-9 * want, "step {index}: {g} against {want}");
        }
        reduced |= index == 2 && got[1] < 0.5 * dsp::summary(&filtered[1]).sd;
    }
    assert!(reduced, "the jolt is taken out");

    // Heart: gathered over several steps of the pulse channel named.
    let config = json!({"openfrenz_model": "pipeline", "step": {"stream": "ppg", "samples": 250}, "nodes": [
        {"name": "pulse", "op": "heart", "in": "ppg", "channel": "tag3", "window_steps": 3}], "outputs": ["pulse"]});
    let mut model = Pipeline::from_dir(model_dir(&config, &[]).path()).unwrap();
    assert_eq!(
        model.spec().outputs,
        ["heart:rate_bpm", "heart:sdnn_ms", "heart:rmssd_ms", "heart:quality"]
    );
    let pulse = |n: usize| 30000.0 + 400.0 * (2.0 * PI * 1.2 * n as f64 / 25.0).sin().max(0.0).powi(4);
    let mut outputs = Vec::new();
    for index in 0..4 {
        let tag3: Vec<f64> = (0..250).map(|n| pulse(index * 250 + n)).collect();
        let block = Block {
            t: vec![0.0; 250],
            values: vec![vec![0.0; 250], tag3],
        };
        outputs.push(model.step(&Step::new(0.0, 0.0, [("ppg".to_owned(), block)])).unwrap());
    }
    assert!(
        outputs[0].is_none() && outputs[1].is_none(),
        "three steps are gathered first"
    );
    let got = outputs[3].as_ref().unwrap();
    let joined: Vec<f64> = (250..1000).map(pulse).collect();
    let want = dsp::heart(&joined, 25.0);
    assert_eq!(got[..], [want.rate_bpm, want.sdnn_ms, want.rmssd_ms, want.quality]);
    assert!(
        (want.rate_bpm - 72.0).abs() < 0.5 && want.quality > 0.95,
        "{} {}",
        want.rate_bpm,
        want.quality
    );
}

#[test]
fn a_stream_the_model_can_do_without_does_not_hold_it_up() {
    let config = json!({"openfrenz_model": "pipeline", "step": {"stream": "eeg", "samples": 250},
        "inputs": {"ppg": {"samples": 50, "required": false}, "battery": {"arrived": true}},
        "nodes": [
            {"name": "size", "op": "summary", "in": "eeg", "stats": ["sd"]},
            {"name": "ppg_slow", "op": "filter", "in": "ppg", "steps": [{"type": "lowpass", "high": 5}]},
            {"name": "pulse", "op": "summary", "in": "ppg_slow", "stats": ["mean"]},
            {"name": "charge", "op": "summary", "in": "battery", "stats": ["mean"]}],
        "outputs": ["size", "pulse", "charge"]});
    let model = Pipeline::from_dir(model_dir(&config, &[]).path()).unwrap();
    let spec = model.spec().clone();
    assert_eq!(spec.outputs.len(), 4 + 2 + 1);
    let input = |stream: &str| spec.inputs.iter().find(|i| i.stream == stream).unwrap().clone();
    assert_eq!(input("ppg"), Input::samples("ppg", 50).preferred());
    assert_eq!(input("battery").take, Take::Arrived);

    // Six steps of EEG; the pulse sensor is silent during the third and fourth, and the
    // battery reports once, during the second.
    let mut stepper = Stepper::new(model).unwrap();
    let times = |from: usize, n: usize, rate: f64| (0..n).map(|k| (from + k) as f64 / rate).collect::<Vec<f64>>();
    stepper.push("battery", &[3.0], &[vec![87.0]], None, false);
    for (from, n) in [(0, 100), (200, 100)] {
        stepper.push("ppg", &times(from, n, 25.0), &[vec![5.0; n], vec![7.0; n]], None, false);
    }
    let eeg: Vec<f64> = (0..1500).map(|k| f64::from(k % 7)).collect();
    stepper.push("eeg", &times(0, 1500, 125.0), &vec![eeg; 4], None, false);
    let outputs = stepper.pop(true).unwrap();
    assert_eq!(
        (outputs.len(), stepper.skipped()),
        (6, 0),
        "no step is lost to the missing pulse"
    );
    let pulse: Vec<bool> = outputs.iter().map(|o| o.values[4].is_finite()).collect();
    assert_eq!(pulse, [true, true, false, false, true, true]);
    assert!(outputs.iter().all(|o| o.values[..4].iter().all(|v| v.is_finite())));
    let charge: Vec<f64> = outputs.iter().map(|o| o.values[6]).collect();
    assert!(
        charge[1] == 87.0 && charge[0].is_nan() && charge[2].is_nan(),
        "{charge:?}"
    );

    // A stream without a steady rate cannot be counted in samples.
    let wrong = json!({"openfrenz_model": "pipeline", "step": {"stream": "eeg", "samples": 250},
        "inputs": {"battery": {"samples": 1}}, "nodes": [], "outputs": []});
    assert!(matches!(Pipeline::from_dir(model_dir(&wrong, &[]).path()), Err(Error::Files(m)) if m.contains("arrived")));
}

#[test]
fn a_config_cannot_ask_for_unbounded_work() {
    // A wavelet is as long as its cycles over its frequency: a long step of a very low
    // frequency would take hours, and is refused when the config is loaded.
    let config = json!({"openfrenz_model": "pipeline", "step": {"stream": "eeg", "samples": 100_000}, "nodes": [
        {"name": "w", "op": "wavelet", "in": "eeg", "frequencies": [0.1], "cycles": 64}], "outputs": ["w"]});
    assert!(
        matches!(Pipeline::from_dir(model_dir(&config, &[]).path()), Err(Error::Files(m)) if m.contains("too much work"))
    );
    // The same node at a sensible size loads.
    let config = json!({"openfrenz_model": "pipeline", "step": {"stream": "eeg", "samples": 250}, "nodes": [
        {"name": "w", "op": "wavelet", "in": "eeg", "frequencies": [2, 6, 10, 20, 40], "cycles": 6}], "outputs": ["w"]});
    assert!(Pipeline::from_dir(model_dir(&config, &[]).path()).is_ok());
}

/// Noise that is the same every run: a fixed sequence between -1 and 1.
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// Four channels of noise; a bump of `height` on both forehead pads, 0.2 s wide, centred on each
/// of `bumps`; and the left forehead pad raised by `shift` from sample `turn` on.
fn eye_signal(samples: usize, bumps: &[usize], height: f64, turn: usize, shift: f64) -> Vec<Vec<f64>> {
    let mut eeg: Vec<Vec<f64>> = (0..4).map(|c| noise(samples, 7 + c)).collect();
    let bump = |n: usize| -> f64 {
        bumps
            .iter()
            .map(|at| height * (-((n as f64 - *at as f64) / 6.0).powi(2)).exp())
            .sum()
    };
    // Channel order of the stream: forehead_right, comb_left, forehead_left, comb_right.
    for (n, x) in eeg[0].iter_mut().enumerate() {
        *x += bump(n);
    }
    for (n, x) in eeg[2].iter_mut().enumerate() {
        *x += bump(n) + if n >= turn { shift } else { 0.0 };
    }
    eeg
}

fn eye_config(samples: usize) -> Value {
    json!({
        "openfrenz_model": "pipeline", "name": "eyes", "version": "1",
        "step": { "stream": "eeg", "samples": samples },
        "nodes": [
            { "name": "pairs", "op": "mix", "in": "eeg", "channels": [
                { "name": "blink", "weights": { "forehead_left": 0.5, "forehead_right": 0.5, "comb_left": -0.5, "comb_right": -0.5 } },
                { "name": "gaze", "weights": { "forehead_left": 1, "forehead_right": -1 } } ] },
            { "name": "b", "op": "select", "in": "pairs", "channels": ["blink"] },
            { "name": "blinks", "op": "peaks", "in": "b", "threshold": 8, "back_s": 1, "warmup_s": 2 },
            { "name": "g", "op": "select", "in": "pairs", "channels": ["gaze"] },
            { "name": "turns", "op": "jumps", "in": "g", "threshold": 8, "leak_s": 4, "warmup_s": 2 }
        ],
        "outputs": ["blinks", "turns"]
    })
}

/// Runs the eye pipeline over `eeg` in steps of `samples`; returns the sample number of every
/// peak, and the position after each sample that ends a step.
fn run_eyes(eeg: &[Vec<f64>], samples: usize) -> (Vec<i64>, Vec<(usize, f64)>, f64) {
    let mut model = Pipeline::from_dir(model_dir(&eye_config(samples), &[]).path()).unwrap();
    assert_eq!(
        model.spec().outputs,
        [
            "blink:peaks",
            "blink:peak_time",
            "blink:peak_size",
            "gaze:position",
            "gaze:jumps"
        ]
    );
    let (mut peaks, mut positions, mut jumps) = (Vec::new(), Vec::new(), 0.0);
    for (k, start) in (0..eeg[0].len()).step_by(samples).enumerate() {
        if start + samples > eeg[0].len() {
            break;
        }
        let block = eeg.iter().map(|c| c[start..start + samples].to_vec()).collect();
        let out = model.step(&eeg_step(block)).unwrap().unwrap();
        let last = (k + 1) * samples - 1;
        if out[0] > 0.0 {
            assert_eq!(out[0], 1.0);
            assert!(out[2] >= 8.0, "a peak below the threshold was reported: {}", out[2]);
            peaks.push(last as i64 + (out[1] * 125.0).round() as i64);
        } else {
            assert!(out[1].is_nan() && out[2].is_nan());
        }
        positions.push((last, out[3]));
        jumps += out[4];
    }
    (peaks, positions, jumps)
}

#[test]
fn peaks_and_jumps_are_found_where_they_were_put() {
    let bumps = [900, 1400, 2130, 2600];
    let eeg = eye_signal(3000, &bumps, 40.0, 1800, 60.0);
    let (peaks, positions, jumps) = run_eyes(&eeg, 25);
    assert_eq!(peaks.len(), bumps.len(), "peaks at {peaks:?}");
    for (found, put) in peaks.iter().zip(bumps) {
        assert!(
            (found - put as i64).abs() <= 3,
            "a peak put at {put} was found at {found}"
        );
    }
    // One turn of 60: nothing before it, all of it soon after, half of it gone 4 s later.
    assert_eq!(jumps, 1.0);
    let at = |sample: usize| positions.iter().find(|(last, _)| *last >= sample).unwrap().1;
    assert_eq!(at(1790), 0.0);
    assert!(
        (at(1840) - 60.0).abs() < 3.0,
        "position just after the turn: {}",
        at(1840)
    );
    assert!((at(1800 + 500) - 30.0).abs() < 2.0, "position 4 s later: {}", at(2300));
}

#[test]
fn peaks_and_jumps_do_not_depend_on_how_the_signal_is_cut_into_steps() {
    let eeg = eye_signal(3000, &[700, 1010, 1600, 2400], 40.0, 1250, -45.0);
    let (peaks, positions, jumps) = run_eyes(&eeg, 25);
    for samples in [10, 125, 250] {
        let (other_peaks, other_positions, other_jumps) = run_eyes(&eeg, samples);
        assert_eq!(peaks, other_peaks, "steps of {samples}");
        assert_eq!(jumps, other_jumps, "steps of {samples}");
        for (last, position) in other_positions {
            if let Some((_, same)) = positions.iter().find(|(l, _)| *l == last) {
                assert!((position - same).abs() < 1e-9, "steps of {samples}, sample {last}");
            }
        }
    }
}

#[test]
fn a_mix_is_a_weighted_sum_and_its_mistakes_are_reported() {
    let config = json!({
        "openfrenz_model": "pipeline", "name": "mix", "version": "1",
        "step": { "stream": "eeg", "samples": 50 },
        "nodes": [
            { "name": "m", "op": "mix", "in": "eeg", "channels": [
                { "name": "difference", "weights": { "forehead_left": 1, "forehead_right": -1 } },
                { "name": "half", "weights": { "comb_left": 0.5 } } ] },
            { "name": "level", "op": "summary", "in": "m", "stats": ["mean", "log_sd"] }
        ],
        "outputs": ["level"]
    });
    let mut model = Pipeline::from_dir(model_dir(&config, &[]).path()).unwrap();
    assert_eq!(
        model.spec().outputs,
        ["difference:mean", "difference:log_sd", "half:mean", "half:log_sd"]
    );
    // forehead_right 1 +- 1, comb_left 4 +- 2, forehead_left 10 +- 1 moving against forehead_right.
    let wave: Vec<f64> = (0..50).map(|n| if n % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let eeg = vec![
        wave.iter().map(|w| 1.0 + w).collect(),
        wave.iter().map(|w| 4.0 + 2.0 * w).collect(),
        wave.iter().map(|w| 10.0 - w).collect(),
        vec![0.0; 50],
    ];
    let out = model.step(&eeg_step(eeg)).unwrap().unwrap();
    let want = [9.0, 2f64.ln(), 2.0, 0.0];
    for (got, want) in out.iter().zip(want) {
        assert!((got - want).abs() < 1e-12, "{out:?}");
    }

    let broken = |change: fn(&mut Value)| {
        let mut config = eye_config(25);
        change(&mut config);
        match Pipeline::from_dir(model_dir(&config, &[]).path()) {
            Err(Error::Files(message)) => message,
            _ => panic!("a broken config was accepted"),
        }
    };
    assert!(broken(|c| c["nodes"][0]["channels"][0]["weights"] = json!({ "nose": 1 })).contains("no channel \"nose\""));
    assert!(broken(|c| c["nodes"][0]["channels"][1]["name"] = json!("blink")).contains("used twice"));
    assert!(broken(|c| c["nodes"][2]["threshold"] = json!(-1)).contains("threshold"));
    assert!(broken(|c| c["nodes"][4]["near_s"] = json!(0.5)).contains("near_s"));
}

#[test]
fn the_models_that_come_with_openfrenz_load_by_name() {
    // Written to the model cache (`~/.cache/openfrenz` or `OPENFRENZ_CACHE`), as for any user.
    let outputs = |name: &str| {
        let dir = frenz_model::resolve(name).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("config.json")).unwrap(),
            frenz_model::BUILT_IN.iter().find(|(n, _)| *n == name).unwrap().1
        );
        Pipeline::from_dir(&dir).unwrap().spec().outputs.clone()
    };
    {
        assert_eq!(
            outputs("eog"),
            [
                "blink:peaks",
                "blink:peak_time",
                "blink:peak_size",
                "gaze_horizontal:position",
                "gaze_horizontal:jumps"
            ]
        );
        assert_eq!(outputs("brow-emg"), ["brow:log_sd"]);
        assert_eq!(
            outputs("heart"),
            ["heart:rate_bpm", "heart:sdnn_ms", "heart:rmssd_ms", "heart:quality"]
        );
        assert_eq!(
            outputs("head"),
            [
                "tilt:pitch_deg",
                "tilt:roll_deg",
                "tilt:from_upright_deg",
                "tilt:g",
                "accel_down:sd",
                "accel_left:sd",
                "accel_forward:sd",
                "turn_shake:sd",
                "turn_nod:sd",
                "turn_side:sd"
            ]
        );
        assert!(frenz_model::resolve("no-such-model").is_err());
    }
    // The same files always have the same fingerprint; any change gives another.
    let (a, b) = (
        frenz_model::BUILT_IN[0].1.as_bytes(),
        frenz_model::BUILT_IN[2].1.as_bytes(),
    );
    assert_eq!(frenz_model::fingerprint(&[a]), frenz_model::fingerprint(&[a]));
    assert_ne!(frenz_model::fingerprint(&[a]), frenz_model::fingerprint(&[b]));
    assert_ne!(
        frenz_model::fingerprint(&[b"ab", b"c"]),
        frenz_model::fingerprint(&[b"a", b"bc"])
    );
}

#[test]
fn tilt_reads_how_the_head_is_held_from_gravity() {
    let config = json!({
        "openfrenz_model": "pipeline", "name": "t", "version": "1",
        "step": { "stream": "accel", "samples": 50 },
        "nodes": [{ "name": "posture", "op": "tilt", "in": "accel" }],
        "outputs": ["posture"]
    });
    let mut model = Pipeline::from_dir(model_dir(&config, &[]).path()).unwrap();
    assert_eq!(
        model.spec().outputs,
        ["tilt:pitch_deg", "tilt:roll_deg", "tilt:from_upright_deg", "tilt:g"]
    );
    let g = 16384.0;
    let mut held = |x: f64, y: f64, z: f64| {
        let block = Block {
            t: vec![0.0; 50],
            values: vec![vec![x * g; 50], vec![y * g; 50], vec![z * g; 50]],
        };
        model
            .step(&Step::new(0.0, 0.0, [("accel".to_owned(), block)]))
            .unwrap()
            .unwrap()
    };
    let near = |got: &[f64], want: [f64; 4], what: &str| {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{what}: {got:?}");
        }
    };
    // x points down, so upright the sensor reads one g against it.
    near(&held(-1.0, 0.0, 0.0), [0.0, 0.0, 0.0, 1.0], "upright");
    let (s, c) = 30f64.to_radians().sin_cos();
    near(&held(-c, 0.0, -s), [30.0, 0.0, 30.0, 1.0], "chin down by 30 degrees");
    near(
        &held(-c, -s, 0.0),
        [0.0, 30.0, 30.0, 1.0],
        "left ear toward the shoulder by 30 degrees",
    );
    near(&held(0.0, 0.0, 1.0), [-90.0, 0.0, 90.0, 1.0], "lying on the back");
    near(&held(0.0, 1.0, 0.0), [0.0, -90.0, 90.0, 1.0], "lying on the right side");
    near(&held(-2.0, 0.0, 0.0), [0.0, 0.0, 0.0, 2.0], "pushed upward");
}
