//! Runs the `frenz` binary end to end on a synthetic recording.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::process::Command;

fn frenz(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_frenz")).args(args).output().unwrap();
    assert!(
        out.status.success(),
        "frenz {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn synth_summary_export() {
    let dir = std::env::temp_dir().join(format!("openfrenz-cli-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let rec = dir.join("demo.sqlite");
    let rec_s = rec.to_str().unwrap();
    assert!(frenz(&["synth", rec_s, "--seconds", "30"]).contains("wrote"));

    let summary = frenz(&["summary", rec_s]);
    assert!(summary.contains("eeg") && summary.contains("gaps 0"), "{summary}");

    let out = dir.join("sessions");
    let exported = frenz(&["export", rec_s, "-o", out.to_str().unwrap()]);
    assert!(exported.contains("Hz"), "{exported}");
    assert!(out.join("demo/eeg.parquet").exists() && out.join("demo/session.json").exists());

    // Refuses to overwrite an existing recording.
    let again = Command::new(env!("CARGO_BIN_EXE_frenz"))
        .args(["synth", rec_s])
        .output()
        .unwrap();
    assert!(!again.status.success());
    std::fs::remove_dir_all(dir).unwrap();
}
