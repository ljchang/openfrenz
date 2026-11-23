//! No band needed: write a synthetic recording and play it back through the live path.
//!
//! `cargo run --example replay`

use frenz::proto::Packet;
use frenz::{Band, Event, ReplayOptions};

#[tokio::main]
async fn main() -> Result<(), frenz::Error> {
    let path = std::env::temp_dir().join(format!("frenz-replay-example-{}.sqlite", std::process::id()));
    frenz::synth::write_recording(&path, 10)?;

    let band = Band::replay(&path, ReplayOptions { speed: 10.0 }).await?;
    let mut events = band.subscribe();
    let mut samples = 0;
    while let Ok(event) = events.recv().await {
        match event {
            Event::Packet { packet, .. } => {
                if let Packet::Eeg(eeg) = packet.as_ref() {
                    samples += eeg.samples.len();
                }
            }
            // A replay ends like a connection that was closed.
            Event::Disconnected { .. } => break,
            _ => {}
        }
    }
    println!("{samples} EEG samples replayed");
    std::fs::remove_file(&path).ok();
    Ok(())
}
