//! The step cutter, the two runners and model loading, with a small test model.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, missing_docs)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use frenz::{Band, ReplayOptions};
use frenz_model::{Error, Input, Live, Model, ModelError, Registry, Spec, Step, Stepper, run_recording};

/// Returns the step number since the last reset, the mean of the first channel of the first
/// stream, and how many samples each other stream contributed (-1 when absent).
struct Probe {
    spec: Spec,
    steps: f64,
    resets: Arc<AtomicU64>,
}

impl Probe {
    fn new(inputs: Vec<Input>) -> Self {
        let mut outputs = vec!["step".to_owned(), "mean".to_owned()];
        outputs.extend(inputs.iter().skip(1).map(|i| format!("{}_samples", i.stream)));
        Self {
            spec: Spec {
                name: "probe".into(),
                version: "1".into(),
                inputs,
                outputs,
                max_step_seconds: 1.1,
            },
            steps: 0.0,
            resets: Arc::default(),
        }
    }
}

impl Model for Probe {
    fn spec(&self) -> &Spec {
        &self.spec
    }
    fn reset(&mut self) {
        self.steps = 0.0;
        self.resets.fetch_add(1, Ordering::Relaxed);
    }
    fn step(&mut self, step: &Step) -> Result<Option<Vec<f64>>, ModelError> {
        let first = step
            .get(&self.spec.inputs[0].stream)
            .ok_or("the first stream is missing")?;
        let x = &first.values[0];
        let mut out = vec![self.steps, x.iter().sum::<f64>() / x.len() as f64];
        for input in self.spec.inputs.iter().skip(1) {
            out.push(step.get(&input.stream).map_or(-1.0, |b| b.t.len() as f64));
        }
        self.steps += 1.0;
        Ok(Some(out))
    }
}

/// `n` samples at `rate` from time `from`, one channel counting up from `from * rate`.
fn ramp(from: f64, n: usize, rate: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let t: Vec<f64> = (0..n).map(|i| from + i as f64 / rate).collect();
    let x = t.iter().map(|t| (t * rate).round()).collect();
    (t, vec![x])
}

#[test]
fn steps_do_not_depend_on_how_samples_are_chunked() {
    let (t, x) = ramp(0.0, 1000, 125.0);
    let mut whole = Stepper::new(Probe::new(vec![Input::samples("eeg", 125)])).unwrap();
    whole.push("eeg", &t, &x, None, false);
    let expected = whole.pop(true).unwrap();
    assert_eq!(expected.len(), 8);
    assert_eq!(expected[3].values, vec![3.0, 375.0 + 62.0]);

    let mut pieces = Stepper::new(Probe::new(vec![Input::samples("eeg", 125)])).unwrap();
    let mut got = Vec::new();
    for (tc, xc) in t.chunks(37).zip(x[0].chunks(37)) {
        pieces.push("eeg", tc, &[xc.to_vec()], None, false);
        got.extend(pieces.pop(false).unwrap());
    }
    got.extend(pieces.pop(true).unwrap());
    assert_eq!(got, expected);
}

#[test]
fn a_new_segment_and_a_hole_restart_the_model() {
    let probe = Probe::new(vec![Input::samples("eeg", 125)]);
    let resets = probe.resets.clone();
    let mut stepper = Stepper::new(probe).unwrap();
    // 300 samples of segment 0, then segment 1: the third step of segment 0 is never complete.
    let (t, x) = ramp(0.0, 300, 125.0);
    stepper.push("eeg", &t, &x, Some(&[0; 300]), false);
    let (t, x) = ramp(10.0, 250, 125.0);
    stepper.push("eeg", &t, &x, Some(&[1; 250]), false);
    // Same segment, but 10 samples lost inside what would be the next step.
    let (t, x) = ramp(12.0, 60, 125.0);
    stepper.push("eeg", &t, &x, Some(&[1; 60]), false);
    let (t, x) = ramp(13.0, 190, 125.0);
    stepper.push("eeg", &t, &x, Some(&[1; 190]), false);
    let out = stepper.pop(true).unwrap();
    let steps: Vec<f64> = out.iter().map(|o| o.values[0]).collect();
    // Two steps, restart at the segment, two steps, one skipped for the hole, one after it.
    assert_eq!(steps, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    assert_eq!(stepper.skipped(), 1);
    assert_eq!(
        stepper.missed().get("eeg"),
        Some(&1),
        "the hole was in the first stream"
    );
    assert_eq!(resets.load(Ordering::Relaxed), 3);
    assert_eq!(out[2].t, 10.0 + 124.0 / 125.0);
}

#[test]
fn a_required_stream_is_needed_and_a_preferred_one_is_not() {
    let (t, x) = ramp(0.0, 375, 125.0);
    let (ta, xa) = ramp(1.0, 50, 50.0); // movement only during the second step
    for (input, expected) in [
        (Input::samples("accel", 50), vec![(0.0, 50.0)]),
        (
            Input::samples("accel", 50).preferred(),
            vec![(0.0, -1.0), (1.0, 50.0), (2.0, -1.0)],
        ),
    ] {
        let mut stepper = Stepper::new(Probe::new(vec![Input::samples("eeg", 125), input])).unwrap();
        stepper.push("accel", &ta, &xa, None, false);
        stepper.push("eeg", &t, &x, None, false);
        stepper.push("gyro", &ta, &xa, None, false); // not asked for: ignored
        let got: Vec<_> = stepper
            .pop(true)
            .unwrap()
            .iter()
            .map(|o| (o.values[0], o.values[2]))
            .collect();
        assert_eq!(got, expected);
        // Only a required stream that is missing costs steps, and it is named.
        let missed = stepper.missed().get("accel").copied().unwrap_or(0);
        assert_eq!(
            (stepper.skipped(), missed),
            if expected.len() == 1 { (2, 2) } else { (0, 0) }
        );
    }
}

#[test]
fn a_step_waits_for_a_late_stream_but_not_for_ever() {
    let inputs = vec![Input::samples("eeg", 125), Input::samples("accel", 50)];
    let mut stepper = Stepper::new(Probe::new(inputs)).unwrap();
    let (t, x) = ramp(0.0, 200, 125.0);
    stepper.push("eeg", &t, &x, None, false);
    assert!(stepper.pop(false).unwrap().is_empty()); // movement may still be on its way
    let (ta, xa) = ramp(0.0, 50, 50.0);
    stepper.push("accel", &ta, &xa, None, false);
    assert_eq!(stepper.pop(false).unwrap().len(), 1);
    // Movement stops arriving: after two steps' worth of waiting the step is given up.
    let (t, x) = ramp(1.6, 300, 125.0);
    stepper.push("eeg", &t, &x, None, false);
    assert!(stepper.pop(false).unwrap().is_empty());
    assert!(stepper.skipped() >= 1);
}

#[test]
fn a_stream_taken_as_it_arrives_is_delivered_once_and_whole() {
    let inputs = vec![Input::samples("eeg", 125), Input::arrived("heart_rate")];
    let mut stepper = Stepper::new(Probe::new(inputs)).unwrap();
    let (t, x) = ramp(0.0, 375, 125.0);
    // One reading inside the first step, one between the first and second, none in the third.
    stepper.push("heart_rate", &[0.5, 0.996, 1.5], &[vec![60.0, 61.0, 62.0]], None, false);
    stepper.push("eeg", &t, &x, None, false);
    let counts: Vec<f64> = stepper.pop(true).unwrap().iter().map(|o| o.values[2]).collect();
    assert_eq!(counts, vec![1.0, 2.0, 0.0]);
}

#[test]
fn a_spec_that_cannot_set_steps_is_refused() {
    for inputs in [
        vec![],
        vec![Input::arrived("battery")],
        vec![Input::samples("eeg", 125).preferred()],
        vec![Input::samples("eeg", 125), Input::samples("eeg", 125)],
    ] {
        assert!(matches!(Stepper::new(Probe::new(inputs)), Err(Error::Spec(_))));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_recording_gives_the_same_steps_offline_and_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("synthetic.sqlite");
    frenz::synth::write_recording_with_pause(&file, 30, Some((12.0, 3.0))).unwrap();

    let inputs = || {
        vec![
            Input::samples("eeg", 125),
            Input::samples("accel", 50),
            Input::samples("ppg", 25).preferred(),
            Input::arrived("battery"),
        ]
    };
    let offline = run_recording(Probe::new(inputs()), &file).unwrap().outputs;
    assert!(offline.len() >= 24, "{} steps", offline.len());
    assert!(offline.iter().all(|o| o.values[2] == 50.0));
    assert!(
        offline.iter().filter(|o| o.values[0] == 0.0).count() >= 2,
        "the pause restarts the model"
    );

    // A replay cuts the same EEG steps: same count, same restart at the pause, same samples.
    // Twice the recorded speed still leaves the pause long enough to be seen.
    let short = dir.path().join("short.sqlite");
    frenz::synth::write_recording_with_pause(&short, 12, Some((4.0, 3.0))).unwrap();
    let eeg_only = || Probe::new(vec![Input::samples("eeg", 125)]);
    let whole = run_recording(eeg_only(), &short).unwrap();
    assert_eq!((whole.skipped, whole.missed.len()), (0, 0));
    let expected = whole.outputs;
    assert!(expected.iter().filter(|o| o.values[0] == 0.0).count() == 2);
    let band = Band::replay(&short, ReplayOptions { speed: 2.0 }).await.unwrap();
    let mut live = Live::new(eeg_only(), &band, false).unwrap();
    let mut got = Vec::new();
    while let Some(output) = live.next().await.unwrap() {
        got.push(output.values);
    }
    assert_eq!(got, expected.into_iter().map(|o| o.values).collect::<Vec<_>>());

    assert!(matches!(
        run_recording(
            Probe::new(vec![Input::samples("eeg", 125), Input::samples("nope", 1)]),
            &file
        ),
        Err(Error::Spec(_))
    ));
}

#[test]
fn model_files_name_the_code_that_loads_them() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), r#"{"openfrenz_model": "probe"}"#).unwrap();
    let registry = Registry::new().with("probe", |_dir| {
        Ok(Box::new(Probe::new(vec![Input::samples("eeg", 125)])) as Box<dyn Model>)
    });
    assert_eq!(registry.load_dir(dir.path()).unwrap().spec().name, "probe");
    let Err(error) = Registry::new().load_dir(dir.path()) else {
        unreachable!("an empty registry loads nothing")
    };
    assert_eq!(
        error.to_string(),
        "needs the model kind \"probe\"; this program has: none"
    );
    std::fs::write(dir.path().join("config.json"), "{}").unwrap();
    assert!(matches!(registry.load_dir(dir.path()), Err(Error::Files(_))));
}
