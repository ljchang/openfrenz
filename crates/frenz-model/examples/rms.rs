//! A small model, run over a recording and over a replay of it.
//!
//!     cargo run -p frenz-model --example rms
#![allow(missing_docs)]

use frenz::{Band, ReplayOptions};
use frenz_model::{Input, Live, Model, ModelError, Spec, Step, run_recording};

/// Root mean square of each EEG channel around its mean, once a second, with the accelerometer's
/// mean vertical reading when the band sends one.
struct Rms(Spec);

impl Rms {
    fn new() -> Self {
        let mut outputs: Vec<String> = frenz_model::streams::channels("eeg")
            .unwrap_or_default()
            .iter()
            .map(|c| format!("rms_{c}"))
            .collect();
        outputs.push("accel_x".into());
        Self(Spec {
            name: "rms".into(),
            version: "1".into(),
            // The first stream sets the steps; the accelerometer is used if it is there.
            inputs: vec![Input::samples("eeg", 125), Input::samples("accel", 50).preferred()],
            outputs,
            max_step_seconds: 1.1,
        })
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}

impl Model for Rms {
    fn spec(&self) -> &Spec {
        &self.0
    }

    fn reset(&mut self) {} // nothing is carried from one step to the next

    fn step(&mut self, step: &Step) -> Result<Option<Vec<f64>>, ModelError> {
        let eeg = step.get("eeg").ok_or("no eeg in the step")?;
        let mut out: Vec<f64> = eeg
            .values
            .iter()
            .map(|x| {
                let m = mean(x);
                mean(&x.iter().map(|v| (v - m).powi(2)).collect::<Vec<_>>()).sqrt()
            })
            .collect();
        let vertical = step
            .get("accel")
            .and_then(|a| a.values.first())
            .map_or(f64::NAN, |x| mean(x));
        out.push(vertical);
        Ok(Some(out))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::temp_dir().join(format!("openfrenz-rms-{}.sqlite", std::process::id()));
    frenz::synth::write_recording(&file, 5)?;

    println!("from the recording:");
    for output in run_recording(Rms::new(), &file)? {
        println!("  {:.2}  {:.1?}", output.t, output.values);
    }

    println!("replayed as a live band:");
    let band = Band::replay(&file, ReplayOptions::default()).await?;
    let mut live = Live::new(Rms::new(), &band, false)?;
    while let Some(output) = live.next().await? {
        println!("  {:.2}  {:.1?}", output.t, output.values);
    }
    std::fs::remove_file(&file)?;
    Ok(())
}
