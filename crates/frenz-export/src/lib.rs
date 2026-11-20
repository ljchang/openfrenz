//! Turns FRENZ recordings into time-aligned, analysis-ready session datasets.
//!
//! A session dataset is a folder with one zstd Parquet file per stream (`eeg`, `accel`, `gyro`,
//! `ppg`, `vitals`, `battery`, `contact`, `status`, `markers`) and a `session.json` describing
//! sources, units, channel map, timing fits and data-quality counts. Every sample row has `t`, its
//! estimated acquisition time in Unix seconds on one host clock, so streams align directly.
//! Values stay in raw device units; nothing is dropped or interpolated. Rows that repeat an earlier
//! block are flagged `duplicate`, and each uninterrupted stretch has its own `segment` number.
//!
//! Source: SQLite recordings written by the `frenz` crate ([`recording`]).

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Map, Value, json};

pub mod recording;
pub mod table;
pub mod timing;

/// Format identifier written to `session.json`.
pub const FORMAT: &str = "openfrenz-session/1";

/// Export errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Reading the SQLite recording failed.
    #[error("recording: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// Filesystem error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Building Arrow data failed.
    #[error("arrow: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    /// Writing Parquet failed.
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// JSON serialization failed.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// The input is not in a recognized or consistent form.
    #[error("invalid input: {0}")]
    Invalid(String),
}

/// An in-memory session: metadata plus one table per stream.
#[derive(Debug, Clone)]
pub struct Session {
    /// `session.json` contents.
    pub meta: Map<String, Value>,
    /// Stream name → table.
    pub tables: BTreeMap<&'static str, table::Table>,
}

impl Session {
    fn new(source: Value) -> Self {
        let mut meta = Map::new();
        meta.insert("format".into(), json!(FORMAT));
        meta.insert(
            "exporter".into(),
            json!(format!("openfrenz frenz-export {}", env!("CARGO_PKG_VERSION"))),
        );
        meta.insert("source".into(), source);
        Self {
            meta,
            tables: BTreeMap::new(),
        }
    }

    /// Writes the dataset folder (created if needed; existing files with the same names are replaced).
    pub fn write(&self, dir: &Path) -> Result<(), Error> {
        std::fs::create_dir_all(dir)?;
        let mut files = Map::new();
        for (name, table) in &self.tables {
            table.write_parquet(&dir.join(format!("{name}.parquet")))?;
            files.insert(
                (*name).to_owned(),
                json!({"file": format!("{name}.parquet"), "rows": table.rows()}),
            );
        }
        let mut meta = self.meta.clone();
        meta.insert("tables".into(), Value::Object(files));
        std::fs::write(
            dir.join("session.json"),
            serde_json::to_string_pretty(&Value::Object(meta))?,
        )?;
        Ok(())
    }
}

/// Reads a `.sqlite` recording.
pub fn read(input: &Path) -> Result<Session, Error> {
    if input.extension().is_some_and(|e| e == "sqlite") {
        recording::from_recording(input)
    } else {
        Err(Error::Invalid(format!("{}: not a .sqlite recording", input.display())))
    }
}

/// EEG channel description shared by all sources (electrode map from the press test).
pub(crate) fn eeg_channel_json() -> Value {
    json!({
        "forehead_right": "slot 0: right forehead pad, beside the optical sensor",
        "comb_left": "slot 1: left side comb (shows eyes-closed alpha)",
        "forehead_left": "slot 3: left forehead pad (strongest blink/EMG)",
        "comb_right": "slot 4: right side comb (shows eyes-closed alpha)",
        "reference": "centre forehead pad (inferred)",
    })
}

/// Accelerometer/gyroscope description.
pub(crate) fn motion_json(accel: bool) -> Value {
    json!({
        "nominal_rate_hz": 50.0,
        "units": "raw signed 16-bit counts",
        "frame": "+x down (toward chin when worn), +y wearer's left, +z forward; y/z rotated ~30° about x",
        "scale": if accel { json!({"counts_per_g_nominal": 16384, "note": "per-unit calibration: see docs/HARDWARE.md"}) } else { json!("unknown") },
    })
}
