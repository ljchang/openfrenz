//! Record everything a band sends to a file until Ctrl-C.
//!
//! `cargo run --example record -- FRENZA12 session.sqlite`

use frenz::{Band, ConnectOptions, Recorder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "FRENZA12".into());
    let path = args.next().unwrap_or_else(|| "session.sqlite".into());

    let band = Band::connect(&name, ConnectOptions::default()).await?;
    // Create the recorder before anything else subscribes, so the file starts at the connection.
    let recorder = Recorder::create(&path, &band)?;
    band.mark("task_start");

    println!("recording to {path}; Ctrl-C to stop");
    tokio::signal::ctrl_c().await?;

    band.mark("task_end");
    let health = band.stop().await?;
    let summary = recorder.finish().await?;
    println!(
        "{} notifications written, integrity {}, {} EEG frames lost",
        summary.rows.0, summary.integrity, health.frames_lost
    );
    Ok(())
}
