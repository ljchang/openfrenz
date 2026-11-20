//! Synthetic recordings: the same file format a real session produces, filled with generated
//! signals. Useful for trying the tools without a band and for end-to-end tests.

// Packet buffers here are allocated at their exact, fixed sizes just before being filled.
#![allow(clippy::indexing_slicing)]

use std::fs::OpenOptions;
use std::path::Path;

use frenz_proto::{Characteristic, Command};
use rusqlite::{Connection, params};

use crate::error::Error;
use crate::record::SCHEMA;

/// Deterministic pseudo-random numbers in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn uuid_text(c: Characteristic) -> String {
    uuid::Uuid::from_u128(c.uuid()).hyphenated().to_string()
}

fn i24(v: i32) -> [u8; 3] {
    let b = v.clamp(-(1 << 23), (1 << 23) - 1).to_be_bytes();
    [b[1], b[2], b[3]]
}

/// Writes a synthetic recording of `seconds` seconds to `path` (which must not exist).
///
/// Streams have their real rates and packet layouts; EEG carries a 10 Hz rhythm on the comb
/// channels plus noise; arrival times include one-sided delivery jitter like a real link.
pub fn write_recording(path: &Path, seconds: u32) -> Result<(), Error> {
    OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut db = Connection::open(path)?;
    db.execute_batch(SCHEMA)?;
    let base_unix: i64 = 1_800_000_000;
    let ns = |t: f64| (t * 1e9) as i64;
    let tx = db.transaction()?;
    {
        let mut meta = tx.prepare("INSERT INTO meta(key, value) VALUES (?1, ?2)")?;
        for (k, v) in [
            ("tool", "openfrenz synthetic recording"),
            ("device_name", "FRENZ-SYNTH"),
            ("firmware", "synthetic"),
        ] {
            meta.execute(params![k, v])?;
        }
        let mut marker = tx.prepare("INSERT INTO markers(mono_ns, wall_ns, label) VALUES (?1, ?2, ?3)")?;
        let mut write =
            tx.prepare("INSERT INTO writes(mono_ns, wall_ns, char, label, payload) VALUES (?1, ?2, ?3, ?4, ?5)")?;
        let mut notif =
            tx.prepare("INSERT INTO notifications(mono_ns, wall_ns, char, payload) VALUES (?1, ?2, ?3, ?4)")?;
        let wall = |mono: i64| base_unix * 1_000_000_000 + mono;
        marker.execute(params![0i64, wall(0), "connected"])?;
        let start = Command::StreamControl {
            mask: 0xff,
            enable: true,
        };
        write.execute(params![
            0i64,
            wall(0),
            uuid_text(Characteristic::CommandRx),
            format!("{start:?}"),
            start.encode().as_bytes()
        ])?;

        // Collect (arrival time, characteristic, payload), then insert in arrival order.
        let mut rng = Lcg(42);
        let mut rows: Vec<(f64, Characteristic, Vec<u8>)> = Vec::new();
        let jitter = |rng: &mut Lcg, scale: f64| 0.004 + scale * rng.next().powi(3);
        let secs_at = |t: f64| ((base_unix as f64 + t) as u32).to_be_bytes();
        let tau = std::f64::consts::TAU;

        let eeg_packets = seconds as usize * 125 / 9;
        for k in 0..eeg_packets {
            let mut p = vec![0u8; 178];
            p[0] = k as u8;
            p[1] = 6;
            let t_last = (9 * k + 8) as f64 / 125.0;
            p[2..6].copy_from_slice(&secs_at(t_last));
            for i in 0..9 {
                let t = (9 * k + i) as f64 / 125.0;
                let alpha = 400.0 * (tau * 10.0 * t).sin();
                let slots = [
                    2_000.0 + 80.0 * (rng.next() - 0.5),
                    140_000.0 + alpha + 120.0 * (rng.next() - 0.5),
                    0.0,
                    12_000.0 + 80.0 * (rng.next() - 0.5),
                    130_000.0 + alpha + 120.0 * (rng.next() - 0.5),
                    0.0,
                ];
                for (j, v) in slots.iter().enumerate() {
                    let at = 6 + 19 * i + 3 * j;
                    p[at..at + 3].copy_from_slice(&i24(*v as i32));
                }
            }
            rows.push((t_last + jitter(&mut rng, 0.12), Characteristic::Eeg, p));
        }
        for (c, gravity) in [
            (Characteristic::Accelerometer, [-15_000.0, -4_000.0, 5_000.0]),
            (Characteristic::Gyroscope, [-100.0, 30.0, -30.0]),
        ] {
            for k in 0..seconds as usize * 2 {
                let mut p = vec![0u8; 158];
                let t_last = (25 * k + 24) as f64 / 50.0;
                p[0..4].copy_from_slice(&secs_at(25.0 * k as f64 / 50.0));
                for i in 0..25 {
                    for (a, g) in gravity.iter().enumerate() {
                        let v = (g + 60.0 * (rng.next() - 0.5)) as i16;
                        let at = 4 + 6 * i + 2 * a;
                        p[at..at + 2].copy_from_slice(&v.to_be_bytes());
                    }
                }
                rows.push((t_last + jitter(&mut rng, 0.12), c, p));
            }
        }
        let mut ppg_n = 0usize;
        for k in 0..seconds as usize * 2 {
            let n = if k % 2 == 0 { 12 } else { 13 };
            let mut p = secs_at(ppg_n as f64 / 25.0).to_vec();
            for i in 0..n {
                let t = (ppg_n + i) as f64 / 25.0;
                let pulse = 600.0 * (tau * 1.1 * t).sin();
                for (tag, v) in [(1u32, 0.0), (3, 135_000.0 + pulse), (2, 83_000.0 + 0.6 * pulse)] {
                    let w = (tag << 19) | ((v as u32) & 0x7_ffff);
                    p.extend_from_slice(&[(w >> 16) as u8, (w >> 8) as u8, w as u8]);
                }
            }
            p.extend_from_slice(&[0, 0]);
            ppg_n += n;
            rows.push((
                (ppg_n - 1) as f64 / 25.0 + jitter(&mut rng, 0.3),
                Characteristic::Ppg,
                p,
            ));
        }
        for s in 0..seconds {
            let t = f64::from(s) + 0.9;
            let secs = secs_at(f64::from(s));
            rows.push((
                t,
                Characteristic::HeartRate,
                [&secs[..], &[1, 64 + (s % 5) as u8]].concat(),
            ));
            rows.push((t, Characteristic::Spo2, [&secs[..], &[1, 97]].concat()));
            rows.push((t + 0.05, Characteristic::Status, vec![0x0e, 0, 0, 0, 0]));
            if s % 5 == 0 {
                rows.push((t, Characteristic::Battery, [&secs[..], &[90]].concat()));
            }
            if s % 2 == 0 {
                let state = match s {
                    0..=3 => 0,
                    4..=7 => 1,
                    _ => 3,
                };
                rows.push((t, Characteristic::Contact, [&secs[..], &[state]].concat()));
            }
        }
        // A Bluetooth link delivers each stream in order: delay may vary but never reorders packets.
        let mut last_arrival = std::collections::HashMap::new();
        for (t, c, _) in &mut rows {
            let last = last_arrival.entry(*c).or_insert(f64::NEG_INFINITY);
            *t = t.max(*last + 1e-4);
            *last = *t;
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (t, c, payload) in &rows {
            notif.execute(params![ns(*t), wall(ns(*t)), uuid_text(*c), payload])?;
        }
        marker.execute(params![
            ns(f64::from(seconds)),
            wall(ns(f64::from(seconds))),
            "disconnected"
        ])?;
    }
    tx.commit()?;
    db.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))?;
    db.query_row("PRAGMA journal_mode=DELETE", [], |_| Ok(()))?;
    Ok(())
}
