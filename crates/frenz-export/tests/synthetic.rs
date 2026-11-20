//! End-to-end: synthetic recording → session → Parquet files.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::path::PathBuf;

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("openfrenz-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn synthetic_recording_exports_with_accurate_clocks() {
    let dir = temp("export");
    let recording = dir.join("synthetic.sqlite");
    frenz::synth::write_recording(&recording, 60).unwrap();
    let session = frenz_export::read(&recording).unwrap();

    for (stream, rows, rate) in [
        ("eeg", 60 * 125 / 9 * 9, 125.0),
        ("accel", 3000, 50.0),
        ("gyro", 3000, 50.0),
        ("ppg", 1500, 25.0),
    ] {
        assert_eq!(session.tables[stream].rows(), rows, "{stream} rows");
        let fits = session.meta["streams"][stream]["segments"].as_array().unwrap();
        assert_eq!(fits.len(), 1, "{stream} should be one uninterrupted segment");
        let fitted = fits[0]["rate_hz"].as_f64().unwrap();
        assert!(
            (fitted - rate).abs() < 0.05 * rate / 125.0 + 0.02,
            "{stream} rate {fitted}"
        );
    }
    for stream in ["vitals", "battery", "contact", "status", "markers"] {
        assert!(session.tables[stream].rows() > 0, "{stream} present");
    }
    assert_eq!(session.meta["health"]["sequence_gaps"], 0);

    let out = dir.join("session");
    session.write(&out).unwrap();
    for stream in session.tables.keys() {
        assert!(out.join(format!("{stream}.parquet")).metadata().unwrap().len() > 0);
    }
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("session.json")).unwrap()).unwrap();
    assert_eq!(meta["format"], frenz_export::FORMAT);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_pause_starts_a_new_segment_even_when_the_counter_continues() {
    let dir = temp("pause");
    let recording = dir.join("paused.sqlite");
    frenz::synth::write_recording_with_pause(&recording, 60, Some((20.0, 5.0))).unwrap();
    let session = frenz_export::read(&recording).unwrap();
    let fits = session.meta["streams"]["eeg"]["segments"].as_array().unwrap();
    assert_eq!(fits.len(), 2, "pause must split the EEG into two segments");
    for fit in fits {
        assert!((fit["rate_hz"].as_f64().unwrap() - 125.0).abs() < 0.3);
        assert!(fit["jitter_p95_s"].as_f64().unwrap() < 0.2);
    }
    std::fs::remove_dir_all(dir).unwrap();
}
