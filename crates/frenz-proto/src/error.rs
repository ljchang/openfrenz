//! Decode errors.

use core::fmt;

use crate::gatt::Characteristic;

/// Why a payload could not be decoded. Carries only wire facts (lengths, byte values), never
/// payload contents, so it is safe to log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// Payload length differs from every verified layout for this characteristic.
    Length {
        /// Characteristic the payload arrived on.
        characteristic: Characteristic,
        /// Observed length in bytes.
        actual: usize,
    },
    /// An EEG frame whose type byte is not the verified value (6). The band has been seen to send
    /// other frame types (for example 106) shortly after connecting; they carry no EEG samples.
    EegFrameType(u8),
    /// An EEG frame without its leading sequence byte (the type byte comes first and the rest is
    /// shifted by one). Seen right after a stream-start command arrived while streams were already
    /// running, carrying a repeat of the preceding block, after which the sequence counter
    /// restarted. Not decoded, so duplicated samples cannot enter a recording.
    EegUnsequenced,
    /// A PPG sample whose three words do not carry tags 1, 2 and 3 exactly once.
    PpgTags {
        /// Index of the offending sample within the packet.
        sample: usize,
    },
    /// A command response with no opcode/status header.
    EmptyResponse,
    /// A command response for an opcode whose payload layout is verified but whose payload has the
    /// wrong length.
    ResponseLength {
        /// Response opcode.
        opcode: u8,
        /// Observed payload length (after opcode and status).
        actual: usize,
    },
    /// A notification on a characteristic this crate does not decode.
    UnknownCharacteristic,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { characteristic, actual } => {
                write!(f, "unsupported {characteristic:?} payload length {actual}")
            }
            Self::EegFrameType(t) => write!(f, "unsupported EEG frame type {t}"),
            Self::EegUnsequenced => write!(f, "EEG frame without sequence byte"),
            Self::PpgTags { sample } => write!(f, "PPG sample {sample} does not carry tags 1, 2 and 3"),
            Self::EmptyResponse => write!(f, "empty command response"),
            Self::ResponseLength { opcode, actual } => {
                write!(f, "response to 0x{opcode:02x} has unexpected payload length {actual}")
            }
            Self::UnknownCharacteristic => write!(f, "notification on an unknown characteristic"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}
