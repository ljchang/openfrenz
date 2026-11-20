//! Client errors.

use std::time::Duration;

/// Errors from connecting to, commanding, or recording a band.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The Bluetooth stack reported an error.
    #[error("bluetooth: {0}")]
    Bluetooth(#[from] btleplug::Error),
    /// No Bluetooth adapter is available (or permission was denied).
    #[error("no Bluetooth adapter available")]
    NoAdapter,
    /// No band with this name was seen within the scan timeout.
    #[error("band {name:?} not found within {timeout:?}; is it on and not connected to another app?")]
    NotFound {
        /// Advertised name searched for.
        name: String,
        /// Scan duration.
        timeout: Duration,
    },
    /// A required characteristic is missing (unsupported firmware or a different device).
    #[error("characteristic {0} missing; unsupported device or firmware")]
    MissingCharacteristic(&'static str),
    /// No response to a command within the timeout.
    #[error("no response to command 0x{opcode:02x} within {timeout:?}")]
    Timeout {
        /// Command opcode.
        opcode: u8,
        /// Time waited.
        timeout: Duration,
    },
    /// The band answered a command with a non-zero status.
    #[error("command 0x{opcode:02x} failed with status {status}")]
    CommandFailed {
        /// Command opcode.
        opcode: u8,
        /// Status byte.
        status: u8,
    },
    /// The connection was lost.
    #[error("disconnected")]
    Disconnected,
    /// Recording storage failed.
    #[error("recording: {0}")]
    Storage(#[from] rusqlite::Error),
    /// Filesystem error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
