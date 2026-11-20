//! `frenz`: scan for, record from, and summarize recordings of FRENZ brainbands.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use frenz::proto::{Characteristic, Packet};
use frenz::{Band, ConnectOptions, Decoder, Event, Health, Recorder, StartSequence};

#[derive(Parser)]
#[command(name = "frenz", version, about = "Unofficial tool for the FRENZ brainband")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List nearby bands.
    Scan {
        /// Seconds to scan.
        #[arg(long, default_value_t = 8.0)]
        seconds: f64,
    },
    /// Record everything a band sends to a SQLite file. Stops on Ctrl-C or after --seconds.
    Record {
        /// Advertised band name, e.g. FRENZA12.
        name: String,
        /// Output file (default: `recordings/NAME-UNIXTIME.sqlite`).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Stop after this many seconds.
        #[arg(long)]
        seconds: Option<f64>,
        /// Start sequence sent after subscribing.
        #[arg(long, value_enum, default_value_t = Start::Standard)]
        start: Start,
        /// Stop when the connection is lost instead of reconnecting.
        #[arg(long)]
        no_reconnect: bool,
        /// Give up reconnecting after this many seconds.
        #[arg(long, default_value_t = 300)]
        reconnect_seconds: u64,
    },
    /// Decode a recording and report stream health.
    Summary {
        /// Recording file.
        file: PathBuf,
    },
    /// Write a synthetic recording (no band needed) for trying the tools.
    Synth {
        /// Output file (must not exist).
        output: PathBuf,
        /// Length in seconds.
        #[arg(long, default_value_t = 60)]
        seconds: u32,
    },
    /// Convert recordings (.sqlite) into time-aligned Parquet session datasets.
    Export {
        /// Recording files.
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output folder; each input becomes <out>/<input name>/.
        #[arg(short, long, default_value = "sessions")]
        out: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Start {
    Standard,
    None,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    match Cli::parse().command {
        Cmd::Scan { seconds } => {
            let found = frenz::scan(Duration::from_secs_f64(seconds)).await?;
            for d in &found {
                println!("{:>5} dBm  {}", d.rssi.map_or("?".into(), |r| r.to_string()), d.name);
            }
            println!("{} band(s) found", found.len());
        }
        Cmd::Record {
            name,
            output,
            seconds,
            start,
            no_reconnect,
            reconnect_seconds,
        } => {
            let reconnect = (!no_reconnect).then(|| Duration::from_secs(reconnect_seconds));
            record(name, output, seconds, start, reconnect).await?
        }
        Cmd::Summary { file } => summary(&file)?,
        Cmd::Synth { output, seconds } => {
            frenz::synth::write_recording(&output, seconds)?;
            println!("wrote {} ({seconds} s synthetic recording)", output.display());
        }
        Cmd::Export { inputs, out } => {
            for input in inputs {
                export(&input, &out).with_context(|| format!("exporting {}", input.display()))?;
            }
        }
    }
    Ok(())
}

fn default_output(name: &str) -> PathBuf {
    let secs = frenz::clock::wall_ns() / 1_000_000_000;
    PathBuf::from("recordings").join(format!("{name}-{secs}.sqlite"))
}

fn print_health(h: &Health) {
    let counts: Vec<String> = h.notifications.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!(
        "  {} | gaps {} (frames lost {}), restarts {}, reconnects {}, irregular EEG {}, decode errors {:?}, battery {:?}%, contact {:?}",
        counts.join(" "),
        h.sequence_gaps,
        h.frames_lost,
        h.counter_restarts,
        h.reconnects,
        h.irregular_eeg_frames,
        h.decode_errors,
        h.battery_percent,
        h.contact_state
    );
}

async fn record(
    name: String,
    output: Option<PathBuf>,
    seconds: Option<f64>,
    start: Start,
    reconnect: Option<Duration>,
) -> Result<()> {
    let path = output.unwrap_or_else(|| default_output(&name));
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let start = match start {
        Start::Standard => StartSequence::Standard,
        Start::None => StartSequence::None,
    };
    println!("connecting to {name} …");
    let band = Band::connect(
        &name,
        ConnectOptions {
            start,
            reconnect,
            ..ConnectOptions::default()
        },
    )
    .await?;
    let info = band.info();
    println!(
        "connected: firmware {:?}, IMU calibration {:?}",
        info.firmware, info.imu_calibration
    );
    let recorder = Recorder::create(&path, &band).with_context(|| format!("creating {}", path.display()))?;
    println!("recording to {} (Ctrl-C to stop)", path.display());

    let mut events = band.subscribe();
    let deadline = seconds.map(|s| tokio::time::Instant::now() + Duration::from_secs_f64(s));
    let mut report = tokio::time::interval(Duration::from_secs(5));
    report.tick().await;
    loop {
        let until_deadline = async {
            match deadline {
                Some(d) => tokio::time::sleep_until(d).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => { println!("stopping …"); break }
            () = until_deadline => break,
            _ = report.tick() => print_health(&band.health()),
            e = events.recv() => match e {
                Ok(Event::Disconnected { .. }) => {
                    println!("  ! connection lost");
                    if reconnect.is_none() { break }
                }
                Ok(Event::Reconnecting { attempt, .. }) => println!("  reconnecting (attempt {attempt}) …"),
                Ok(Event::Reconnected { .. }) => println!("  reconnected; streaming again"),
                Ok(Event::ReconnectFailed { .. }) => { println!("  ! gave up reconnecting"); break }
                Ok(Event::SequenceGap { expected, received, .. }) => println!("  ! EEG gap: expected {expected}, got {received}"),
                Ok(Event::OffHead { .. }) => println!("  ! band reports no contact (it powers off ~33 s later)"),
                Ok(Event::OnHead { .. }) => println!("  contact restored"),
                Ok(Event::PowerOffWarning { .. }) => println!("  ! power-off warning"),
                Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
        }
    }
    let health = band.stop().await?;
    let summary = recorder.finish().await?;
    print_health(&health);
    println!(
        "wrote {}: {} notifications, {} writes, {} markers; recorder lagged {}; integrity {}",
        summary.path.display(),
        summary.rows.0,
        summary.rows.1,
        summary.rows.2,
        summary.lagged,
        summary.integrity
    );
    Ok(())
}

fn summary(file: &PathBuf) -> Result<()> {
    let db = rusqlite::Connection::open_with_flags(file, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let starts: Vec<i64> = db
        .prepare("SELECT mono_ns FROM writes WHERE hex(payload) = '03FF01' ORDER BY id")?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    let mut stmt = db.prepare("SELECT mono_ns, char, payload FROM notifications ORDER BY id")?;
    let mut rows = stmt.query([])?;
    let (mut decoder, mut health, mut out) = (Decoder::default(), Health::default(), Vec::new());
    let (mut first, mut last) = (i64::MAX, i64::MIN);
    let mut samples = std::collections::BTreeMap::<&str, u64>::new();
    while let Some(row) = rows.next()? {
        let t: i64 = row.get(0)?;
        let uuid: String = row.get(1)?;
        let payload: Vec<u8> = row.get(2)?;
        let Some(c) = uuid::Uuid::parse_str(&uuid)
            .ok()
            .and_then(|u| Characteristic::from_uuid(u.as_u128()))
        else {
            continue;
        };
        let last_start = starts.iter().rev().find(|s| **s <= t).copied().unwrap_or(i64::MIN / 2);
        decoder.handle(t, c, &payload, last_start, &mut health, &mut out);
        for e in out.drain(..) {
            if let Event::Packet { packet, .. } = e {
                let (name, n) = match packet.as_ref() {
                    Packet::Eeg(_) => ("eeg", 9),
                    Packet::Accelerometer(_) => ("accelerometer", 25),
                    Packet::Gyroscope(_) => ("gyroscope", 25),
                    Packet::Ppg(p) => ("ppg", p.samples().len() as u64),
                    _ => continue,
                };
                *samples.entry(name).or_default() += n;
            }
        }
        first = first.min(t);
        last = last.max(t);
    }
    let span = (last - first) as f64 / 1e9;
    println!("{}: {:.1} s", file.display(), span);
    for (name, n) in &samples {
        println!("  {name:>13}: {n:>7} samples, {:.2} Hz", *n as f64 / span);
    }
    print_health(&health);
    Ok(())
}

fn export(input: &std::path::Path, out: &std::path::Path) -> Result<()> {
    let session = frenz_export::read(input)?;
    let name = input
        .file_stem()
        .map_or_else(|| "session".into(), |s| s.to_string_lossy().to_string());
    let dir = out.join(name);
    session.write(&dir)?;
    println!("{} -> {}", input.display(), dir.display());
    for (stream, table) in &session.tables {
        let info = session.meta.get("streams").and_then(|s| s.get(*stream));
        let rates: Vec<String> = info
            .and_then(|i| i.get("segments"))
            .and_then(serde_json::Value::as_array)
            .map(|segs| {
                segs.iter()
                    .filter_map(|s| {
                        Some(format!(
                            "{:.3} Hz (jitter p95 {:.0} ms)",
                            s.get("rate_hz")?.as_f64()?,
                            s.get("jitter_p95_s")?.as_f64()? * 1e3
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        println!("  {stream:>8}: {:>7} rows {}", table.rows(), rates.join(", "));
    }
    Ok(())
}
