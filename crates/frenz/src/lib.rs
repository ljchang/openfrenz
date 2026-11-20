//! Async client for the FRENZ brainband over Bluetooth LE.
//!
//! ```no_run
//! # async fn demo() -> Result<(), frenz::Error> {
//! use frenz::{Band, ConnectOptions, Event};
//! use frenz::proto::Packet;
//!
//! let band = Band::connect("FRENZA12", ConnectOptions::default()).await?;
//! let mut events = band.subscribe();
//! while let Ok(event) = events.recv().await {
//!     if let Event::Packet { packet, .. } = event {
//!         if let Packet::Eeg(eeg) = packet.as_ref() {
//!             println!("{:?}", eeg.samples[0]);
//!         }
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! Notifications are timestamped on arrival and handed to a decoder task; nothing on the
//! Bluetooth callback path parses or blocks. Every irregularity (decode error, sequence gap,
//! counter restart, lagging subscriber, disconnect) is an explicit [`Event`] and is counted in
//! [`Health`]. Nothing is deduplicated, reordered, or invented.

pub mod band;
pub mod clock;
pub mod error;
pub mod event;
pub mod record;
pub mod synth;

pub use band::{Band, BandInfo, ConnectOptions, Discovered, StartSequence, scan};
pub use error::Error;
pub use event::{Decoder, Event, Health, RawRecord};
pub use frenz_proto as proto;
pub use record::{RecordSummary, Recorder};
