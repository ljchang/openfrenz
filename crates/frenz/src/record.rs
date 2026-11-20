//! Lossless, crash-safe recording of everything the band sends, to one SQLite file per session.
//!
//! Durability: exclusive creation (never overwrite),
//! WAL with `synchronous=FULL` and `fullfsync`, batched transactions of at most 100 ms or 4096 rows,
//! and on finish an integrity check, checkpoint and switch to a rollback journal so the result is a
//! single self-contained file.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rusqlite::{Connection, params};
use tokio::sync::{broadcast, oneshot};

use crate::band::Band;
use crate::clock::{anchor_wall_ns, mono_ns, wall_ns};
use crate::error::Error;
use crate::event::RawRecord;

pub(crate) const SCHEMA: &str = "
    PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA fullfsync=ON; PRAGMA checkpoint_fullfsync=ON;
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE gatt (service TEXT, char TEXT, handle INTEGER, properties TEXT, descriptors TEXT);
    CREATE TABLE notifications (id INTEGER PRIMARY KEY, mono_ns INTEGER NOT NULL, wall_ns INTEGER NOT NULL,
        char TEXT NOT NULL, payload BLOB NOT NULL);
    CREATE TABLE writes (id INTEGER PRIMARY KEY, mono_ns INTEGER NOT NULL, wall_ns INTEGER NOT NULL,
        char TEXT NOT NULL, label TEXT NOT NULL, payload BLOB NOT NULL, error TEXT);
    CREATE TABLE markers (id INTEGER PRIMARY KEY, mono_ns INTEGER NOT NULL, wall_ns INTEGER NOT NULL, label TEXT NOT NULL);
";

/// Outcome of a finished recording.
#[derive(Debug, Clone)]
pub struct RecordSummary {
    /// File written.
    pub path: PathBuf,
    /// Rows written per table: notifications, writes, markers.
    pub rows: (u64, u64, u64),
    /// Raw records the recorder missed because it fell behind (should always be 0).
    pub lagged: u64,
    /// Result of `PRAGMA integrity_check` ("ok" when sound).
    pub integrity: String,
}

/// Records a band's raw traffic until [`Recorder::finish`].
pub struct Recorder {
    path: PathBuf,
    stop: Option<oneshot::Sender<()>>,
    forwarder: tokio::task::JoinHandle<u64>,
    writer: JoinHandle<Result<(u64, u64, u64), Error>>,
}

fn uuid_text(c: frenz_proto::Characteristic) -> String {
    uuid::Uuid::from_u128(c.uuid()).hyphenated().to_string()
}

impl Recorder {
    /// Creates `path` (which must not exist) and starts recording `band`.
    pub fn create(path: impl AsRef<Path>, band: &Band) -> Result<Self, Error> {
        let path = path.as_ref().to_path_buf();
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?
            .sync_all()?;
        let db = Connection::open(&path)?;
        db.execute_batch(SCHEMA)?;
        let info = band.info();
        let meta = [
            ("tool", format!("openfrenz frenz {}", env!("CARGO_PKG_VERSION"))),
            ("device_name", info.name.clone()),
            ("firmware", info.firmware.clone().unwrap_or_default()),
            ("imu_calibration", format!("{:?}", info.imu_calibration)),
            (
                "clock_anchor",
                serde_json::json!({"mono_ns": 0, "wall_ns": anchor_wall_ns()}).to_string(),
            ),
            (
                "clock_pair_at_start",
                serde_json::json!({"mono_ns": mono_ns(), "wall_ns": wall_ns()}).to_string(),
            ),
        ];
        for (k, v) in meta {
            db.execute("INSERT INTO meta(key, value) VALUES (?1, ?2)", params![k, v])?;
        }
        for (service, characteristic, props) in &info.gatt {
            db.execute(
                "INSERT INTO gatt(service, char, properties) VALUES (?1, ?2, ?3)",
                params![service, characteristic, props],
            )?;
        }

        let (tx, rx) = mpsc::channel::<RawRecord>();
        let writer = std::thread::Builder::new()
            .name("frenz-recorder".into())
            .spawn(move || write_loop(db, rx))?;
        let (stop, mut stop_rx) = oneshot::channel();
        let mut raw = band.subscribe_raw_from_start();
        let forwarder = tokio::spawn(async move {
            let mut lagged = 0u64;
            loop {
                tokio::select! {
                    biased;
                    r = raw.recv() => match r {
                        Ok(record) => if tx.send(record).is_err() { break },
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            lagged += n;
                            let label = format!("recorder_lagged:{n}");
                            let _ = tx.send(RawRecord::Marker { mono_ns: mono_ns(), wall_ns: wall_ns(), label });
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    },
                    _ = &mut stop_rx => {
                        // Drain what has already arrived, then stop.
                        while let Ok(record) = raw.try_recv() {
                            let _ = tx.send(record);
                        }
                        break;
                    }
                }
            }
            lagged
        });
        Ok(Self {
            path,
            stop: Some(stop),
            forwarder,
            writer,
        })
    }

    /// Stops recording, commits everything received, verifies and finalizes the file.
    pub async fn finish(mut self) -> Result<RecordSummary, Error> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let lagged = self.forwarder.await.unwrap_or_default();
        let writer = self.writer;
        let rows = tokio::task::spawn_blocking(move || writer.join())
            .await
            .map_err(|e| std::io::Error::other(e.to_string()))?
            .map_err(|_| std::io::Error::other("recorder thread panicked"))??;
        let db = Connection::open(&self.path)?;
        let integrity: String = db.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
        db.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('clock_pair_at_end', ?1)",
            params![serde_json::json!({"mono_ns": mono_ns(), "wall_ns": wall_ns()}).to_string()],
        )?;
        db.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))?;
        db.query_row("PRAGMA journal_mode=DELETE", [], |_| Ok(()))?;
        Ok(RecordSummary {
            path: self.path,
            rows,
            lagged,
            integrity,
        })
    }
}

fn write_loop(mut db: Connection, rx: mpsc::Receiver<RawRecord>) -> Result<(u64, u64, u64), Error> {
    let mut rows = (0u64, 0u64, 0u64);
    let mut open = true;
    while open {
        let mut batch = Vec::new();
        let deadline = Instant::now() + Duration::from_millis(100);
        while batch.len() < 4096 {
            match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(r) => batch.push(r),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    open = false;
                    break;
                }
            }
        }
        if batch.is_empty() {
            continue;
        }
        let tx = db.transaction()?;
        {
            let mut notif = tx
                .prepare_cached("INSERT INTO notifications(mono_ns, wall_ns, char, payload) VALUES (?1, ?2, ?3, ?4)")?;
            let mut write = tx.prepare_cached(
                "INSERT INTO writes(mono_ns, wall_ns, char, label, payload, error) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let mut marker = tx.prepare_cached("INSERT INTO markers(mono_ns, wall_ns, label) VALUES (?1, ?2, ?3)")?;
            for r in &batch {
                match r {
                    RawRecord::Notification {
                        mono_ns,
                        wall_ns,
                        characteristic,
                        bytes,
                    } => {
                        notif.execute(params![mono_ns, wall_ns, uuid_text(*characteristic), &bytes[..]])?;
                        rows.0 += 1;
                    }
                    RawRecord::Write {
                        mono_ns,
                        wall_ns,
                        label,
                        bytes,
                        error,
                    } => {
                        let c = uuid_text(frenz_proto::Characteristic::CommandRx);
                        write.execute(params![mono_ns, wall_ns, c, label, &bytes[..], error])?;
                        rows.1 += 1;
                    }
                    RawRecord::Marker {
                        mono_ns,
                        wall_ns,
                        label,
                    } => {
                        marker.execute(params![mono_ns, wall_ns, label])?;
                        rows.2 += 1;
                    }
                }
            }
        }
        tx.commit()?;
    }
    Ok(rows)
}
