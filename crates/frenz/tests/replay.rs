//! Replaying a synthetic recording drives the live event path without a band.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use frenz::proto::{Command, Packet};
use frenz::{Band, Error, Event, Recorder, ReplayOptions};

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("openfrenz-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn replay_delivers_every_packet_then_disconnects() {
    let dir = temp("replay");
    let source = dir.join("synthetic.sqlite");
    frenz::synth::write_recording(&source, 10).unwrap();

    let band = Band::replay(&source, ReplayOptions { speed: 0.0 }).await.unwrap();
    assert_eq!(band.info().name, "FRENZ-SYNTH");
    assert_eq!(band.info().firmware.as_deref(), Some("synthetic"));
    let recorder = Recorder::create(dir.join("copy.sqlite"), &band).unwrap();
    let mut events = band.subscribe();

    let (mut eeg, mut motion, mut ppg) = (0, 0, 0);
    loop {
        match tokio::time::timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap()
        {
            Event::Packet { packet, .. } => match packet.as_ref() {
                Packet::Eeg(_) => eeg += 1,
                Packet::Accelerometer(_) | Packet::Gyroscope(_) => motion += 1,
                Packet::Ppg(_) => ppg += 1,
                _ => {}
            },
            Event::Disconnected { .. } => break,
            Event::SequenceGap { .. } | Event::DecodeError { .. } => panic!("synthetic data has no gaps or errors"),
            _ => {}
        }
    }
    assert_eq!((eeg, motion, ppg), (10 * 125 / 9, 40, 20));

    let health = band.health();
    assert!(!health.connected);
    assert_eq!((health.sequence_gaps, health.frames_lost), (0, 0));
    assert_eq!(health.battery_percent, Some(90));

    // There is no device behind a replay.
    assert!(matches!(band.send(Command::GetBattery).await, Err(Error::Disconnected)));

    // A recorder attached to a replay captures every notification, like a live one.
    let summary = recorder.finish().await.unwrap();
    let total: u64 = health.notifications.values().sum();
    assert_eq!(summary.rows.0, total);
    assert_eq!(summary.integrity, "ok");
}

#[tokio::test]
async fn replay_keeps_recorded_pacing() {
    let dir = temp("replay-paced");
    let source = dir.join("synthetic.sqlite");
    frenz::synth::write_recording(&source, 4).unwrap();

    // 4 s of data at 8x should take about half a second.
    let band = Band::replay(&source, ReplayOptions { speed: 8.0 }).await.unwrap();
    let mut events = band.subscribe();
    let started = Instant::now();
    loop {
        if let Event::Disconnected { .. } = tokio::time::timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap()
        {
            break;
        }
    }
    let took = started.elapsed().as_secs_f64();
    assert!((0.35..1.5).contains(&took), "took {took} s");
}

/// A band restarts its frame counter at 0 when streams are started again (after a reconnect).
/// A replay must reproduce the recorded start command, or the restart looks like lost frames.
#[tokio::test]
async fn replay_reproduces_stream_starts() {
    let dir = temp("replay-start");
    let source = dir.join("synthetic.sqlite");
    frenz::synth::write_recording(&source, 6).unwrap();
    let (writes_before, restart_at) = {
        let db = rusqlite::Connection::open(&source).unwrap();
        let eeg: Vec<(i64, i64, Vec<u8>)> = db
            .prepare("SELECT id, mono_ns, payload FROM notifications WHERE length(payload) = 178 ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        // From the middle of the recording on, the counter starts again at 0, just after a start
        // command.
        let half = eeg.len() / 2;
        let restart_at = eeg[half].1;
        for (count, (id, _, payload)) in eeg[half..].iter().enumerate() {
            let mut payload = payload.clone();
            payload[0] = count as u8;
            db.execute("UPDATE notifications SET payload = ?1 WHERE id = ?2", (payload, id))
                .unwrap();
        }
        db.execute(
            "INSERT INTO writes(mono_ns, wall_ns, char, label, payload) VALUES (?1, 0, 'command_rx', 'restart', x'03FF01')",
            [restart_at - 1_000_000],
        )
        .unwrap();
        let writes: i64 = db.query_row("SELECT count(*) FROM writes", [], |r| r.get(0)).unwrap();
        (writes, restart_at)
    };
    assert!(restart_at > 0);

    // Paced, so that the restart comes more than a second after playback began: the start of a
    // replay counts as a stream start itself, and would explain a restart that soon after it.
    let band = Band::replay(&source, ReplayOptions { speed: 2.0 }).await.unwrap();
    let recorder = Recorder::create(dir.join("copy.sqlite"), &band).unwrap();
    let mut events = band.subscribe();
    while !matches!(
        tokio::time::timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap(),
        Event::Disconnected { .. }
    ) {}
    let health = band.health();
    assert_eq!(
        (health.counter_restarts, health.sequence_gaps, health.frames_lost),
        (1, 0, 0)
    );

    // A recording of the replay holds the same commands as the original.
    let summary = recorder.finish().await.unwrap();
    assert_eq!(summary.rows.1, writes_before as u64);
    std::fs::remove_dir_all(&dir).ok();
}
