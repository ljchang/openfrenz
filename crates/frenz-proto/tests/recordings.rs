//! Decodes every packet of every local recording (if any) and checks stream invariants.
//!
//! Recordings hold human physiological data and are never committed. This test runs against
//! `$OPENFRENZ_RECORDINGS` (default: `<repo>/recordings`) and is a no-op when it is absent.
//! It reads the SQLite files with the `sqlite3` command-line tool to keep this crate dependency-free.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

use std::path::{Path, PathBuf};
use std::process::Command;

use frenz_proto::{Characteristic, Packet, ParseError};

/// The stream-start command restarts the EEG sequence counter at 0 when it arrives while streams
/// are already running (observed). Such restarts within this window of the command are expected.
const RESTART_WINDOW_NS: i64 = 1_000_000_000;

fn recordings() -> Vec<PathBuf> {
    let dir = std::env::var_os("OPENFRENZ_RECORDINGS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../recordings"));
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "sqlite"))
        .collect();
    files.sort();
    files
}

fn query(db: &Path, sql: &str) -> Vec<Vec<String>> {
    let out = Command::new("sqlite3")
        .arg("-readonly")
        .arg(db)
        .arg(sql)
        .output()
        .expect("sqlite3 CLI available");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.split('|').map(str::to_owned).collect())
        .collect()
}

fn unhex(h: &str) -> Vec<u8> {
    (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex"))
        .collect()
}

#[test]
fn every_recorded_packet_decodes() {
    let files = recordings();
    if files.is_empty() {
        eprintln!("no local recordings; skipping");
        return;
    }
    for db in files {
        let starts: Vec<i64> = query(&db, "SELECT mono_ns FROM writes WHERE hex(payload) = '03FF01'")
            .into_iter()
            .map(|r| r[0].parse().expect("ns"))
            .collect();
        let (mut decoded, mut irregular, mut restarts, mut gaps) = (0usize, 0usize, 0usize, 0usize);
        let mut eeg_sequence: Option<u8> = None;
        let mut errors = Vec::new();
        for row in query(&db, "SELECT mono_ns, char, hex(payload) FROM notifications ORDER BY id") {
            let t: i64 = row[0].parse().expect("ns");
            let c = Characteristic::from_uuid(u128::from_str_radix(&row[1].replace('-', ""), 16).expect("uuid"))
                .expect("only FRENZ characteristics are recorded");
            match Packet::decode(c, &unhex(&row[2])) {
                Ok(Packet::Eeg(e)) => {
                    assert!(
                        e.samples.iter().all(|s| s[2] == 0 && s[5] == 0),
                        "inactive slots are zero"
                    );
                    if let Some(prev) = eeg_sequence
                        && e.sequence != prev.wrapping_add(1)
                    {
                        let after_start = starts.iter().any(|s| (0..RESTART_WINDOW_NS).contains(&(t - s)));
                        if e.sequence == 0 && after_start {
                            restarts += 1
                        } else {
                            gaps += 1
                        }
                    }
                    eeg_sequence = Some(e.sequence);
                    decoded += 1;
                }
                Ok(_) => decoded += 1,
                Err(ParseError::EegFrameType(_) | ParseError::EegUnsequenced) => {
                    irregular += 1;
                    eeg_sequence = None;
                }
                Err(e) => errors.push((c, e)),
            }
        }
        println!(
            "{}: {decoded} decoded, {irregular} irregular EEG frames, {restarts} counter restarts, {gaps} gaps",
            db.file_name().unwrap().to_string_lossy()
        );
        assert!(
            errors.is_empty(),
            "{}: {} decode errors, first {:?}",
            db.display(),
            errors.len(),
            errors.first()
        );
        assert_eq!(gaps, 0, "{}", db.display());
    }
}
