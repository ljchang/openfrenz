//! Connect to a band and print live EEG and what happens on the link.
//!
//! `cargo run --example stream -- FRENZA12`

use frenz::proto::Packet;
use frenz::{Band, ConnectOptions, Event};

#[tokio::main]
async fn main() -> Result<(), frenz::Error> {
    let name = std::env::args().nth(1).unwrap_or_else(|| "FRENZA12".into());
    let band = Band::connect(&name, ConnectOptions::default()).await?;
    println!(
        "connected: {:?}, battery {} %",
        band.info().firmware,
        band.battery().await?
    );

    let mut events = band.subscribe();
    let mut frames = 0;
    while frames < 500 {
        match events.recv().await {
            Ok(Event::Packet { packet, .. }) => {
                if let Packet::Eeg(eeg) = packet.as_ref() {
                    // Nine samples per frame; slots 0, 1, 3 and 4 hold the four electrodes.
                    let [fr, cl, _, fl, cr, _] = eeg.samples[0];
                    println!("{:>3} {fr:>8} {cl:>8} {fl:>8} {cr:>8}", eeg.sequence);
                    frames += 1;
                }
            }
            Ok(Event::SequenceGap { expected, received, .. }) => {
                println!("EEG frames lost: expected {expected}, received {received}");
            }
            Ok(Event::OffHead { .. }) => println!("no electrode contact"),
            Ok(Event::Disconnected { .. }) => println!("connection lost; reconnecting"),
            Ok(Event::Reconnected { .. }) => println!("reconnected"),
            Ok(Event::ReconnectFailed { .. }) => break,
            Ok(_) => {}
            // This subscriber fell behind and `n` events were skipped for it.
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => println!("skipped {n} events"),
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
        }
    }

    let health = band.stop().await?;
    println!("gaps {}, frames lost {}", health.sequence_gaps, health.frames_lost);
    Ok(())
}
