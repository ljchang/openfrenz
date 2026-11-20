//! Wire protocol of the FRENZ brainband: characteristic identifiers, stream decoders, command
//! encoders and command-response decoders.
//!
//! This crate performs no I/O and allocates nothing. Every decoder is strict: a payload whose
//! length or structure differs from the verified layout returns a [`ParseError`] instead of being
//! guessed at, and no input can cause a panic. Fields whose meaning is not yet known are kept raw.
//!
//! The layouts and their verification status are documented in `docs/PROTOCOL.md`.
#![cfg_attr(not(feature = "std"), no_std)]

pub mod command;
pub mod error;
pub mod gatt;
pub mod stream;

pub use command::{Command, Response, ResponseBody};
pub use error::ParseError;
pub use gatt::Characteristic;
pub use stream::{
    BatteryPacket, ContactPacket, EegPacket, MotionPacket, Packet, PpgPacket, StatusMessage, VitalPacket,
};

/// Reads a big-endian `u32` at `offset`, or `None` if out of range.
pub(crate) fn be_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let b = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_be_bytes([*b.first()?, *b.get(1)?, *b.get(2)?, *b.get(3)?]))
}

/// Reads a big-endian `u16` at `offset`, or `None` if out of range.
pub(crate) fn be_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_be_bytes([
        *bytes.get(offset)?,
        *bytes.get(offset.checked_add(1)?)?,
    ]))
}
