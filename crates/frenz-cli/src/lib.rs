//! The `frenz` command: scan for, record from, summarize and export recordings of FRENZ brainbands.
//!
//! The command is a library function so that the `frenz` binary and the Python package's `frenz`
//! entry point run the same code.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use frenz::{Band, ConnectOptions, Event, Health, Recorder, StartSequence};

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
        /// Also run a model over each recording and write what it gives as a table: a model
        /// that comes with openfrenz (eog, brow-emg, heart, head), a folder, or a Hugging Face repository
        /// (owner/name). Repeat for several.
        #[arg(short, long = "model")]
        models: Vec<String>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Start {
    Standard,
    None,
}

/// Runs the `frenz` command with these arguments (the first is the program name) and returns the
/// process exit code. Help, usage errors and failures are printed.
pub fn run<I, T>(args: I) -> u8
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(e) => {
            e.print().ok();
            return u8::try_from(e.exit_code()).unwrap_or(2);
        }
    };
    // A second call in one process (possible from Python) keeps the first subscriber.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .ok();
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    match runtime.block_on(dispatch(cli)) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {e:?}");
            1
        }
    }
}

async fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
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
        Cmd::Export { inputs, out, models } => {
            for input in inputs {
                export(&input, &out, &models).with_context(|| format!("exporting {}", input.display()))?;
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

fn summary(file: &std::path::Path) -> Result<()> {
    let summary = frenz_export::recording::summarize(file)?;
    println!("{}: {:.1} s", file.display(), summary.seconds);
    for (name, n) in &summary.samples {
        println!("  {name:>13}: {n:>7} samples, {:.2} Hz", *n as f64 / summary.seconds);
    }
    print_health(&summary.health);
    Ok(())
}

fn export(input: &std::path::Path, out: &std::path::Path, models: &[String]) -> Result<()> {
    let session = frenz_export::read(input)?;
    let name = input
        .file_stem()
        .map_or_else(|| "session".into(), |s| s.to_string_lossy().to_string());
    let dir = out.join(name);
    session.write(&dir)?;
    let modelled = frenz_model::export_models(&session, &dir, models)?;
    println!("{} -> {}", input.display(), dir.display());
    for made in &modelled {
        let missed: Vec<String> = made
            .missed
            .iter()
            .map(|(stream, n)| format!("{n} without {stream}"))
            .collect();
        let skipped = if missed.is_empty() {
            String::new()
        } else {
            format!(" (steps skipped: {})", missed.join(", "))
        };
        println!("  {:>12}: {} steps from the model{skipped}", made.table, made.rows);
    }
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
