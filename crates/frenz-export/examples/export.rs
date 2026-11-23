//! Turn a recording into a session folder and print what it holds.
//!
//! `cargo run --example export -- session.sqlite sessions/session`

use std::path::Path;

fn main() -> Result<(), frenz_export::Error> {
    let mut args = std::env::args().skip(1);
    let recording = args.next().unwrap_or_else(|| "session.sqlite".into());
    let out = args.next().unwrap_or_else(|| "sessions/session".into());

    // Duration, sample counts and health counters, without exporting.
    let summary = frenz_export::recording::summarize(Path::new(&recording))?;
    println!(
        "{:.1} s, {} EEG frames lost",
        summary.seconds, summary.health.frames_lost
    );

    // One Parquet file per stream plus session.json.
    let session = frenz_export::read(Path::new(&recording))?;
    session.write(Path::new(&out))?;
    for (stream, table) in &session.tables {
        println!("{stream:>8}: {} rows", table.rows());
    }
    Ok(())
}
