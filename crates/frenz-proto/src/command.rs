//! Commands (written to [`Characteristic::CommandRx`](crate::Characteristic::CommandRx)) and their
//! responses (notified on [`Characteristic::CommandResponse`](crate::Characteristic::CommandResponse)).
//!
//! Only commands whose effect has been observed on hardware are exposed. Opcodes for audio,
//! alarms, stimulation, file transfer, firmware update, test modes, reset and renaming exist on the
//! device but are deliberately absent until they have been characterized on a bench.

use crate::error::ParseError;
use crate::{be_u16, be_u32};

/// A command the client may send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Request the firmware version string.
    GetFirmwareVersion,
    /// Request the battery percentage.
    GetBattery,
    /// Request the serial number.
    GetSerialNumber,
    /// Set the device clock (Unix seconds). Afterwards packet device times are Unix seconds.
    TimeSync(u32),
    /// Request the advertised device name.
    GetDeviceName,
    /// Request the device clock.
    GetDeviceTime,
    /// Request the device identifier (the Bluetooth address as text).
    GetDeviceId,
    /// Enable or disable periodic heart-rate/SpO2 reporting (1 Hz when enabled, every 5 s otherwise).
    HeartRatePeriodic(bool),
    /// Session-start notice (`0e 00`), part of the standard start sequence.
    OnConnected,
    /// Start or stop streaming. `mask` is sent as-is; `0xff` (all streams) is the only verified value.
    StreamControl {
        /// Stream selection mask.
        mask: u8,
        /// Start (`true`) or stop.
        enable: bool,
    },
    /// Set the volume of the band's spoken prompts (`0` = silent).
    SetPromptVolume(u8),
    /// Request the IMU calibration report.
    GetImuCalibration,
    /// Session-running notice (`0e 02`), part of the standard start sequence. Without it the band
    /// powers itself off 10 minutes after streaming starts.
    StartSession,
    /// Session-end notice (`0e 03`), part of the standard stop sequence; effect not characterized.
    EndSession,
}

/// An encoded command, at most 8 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Encoded {
    buf: [u8; 8],
    len: u8,
}

impl Encoded {
    fn new(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 8];
        let len = bytes.len().min(8);
        if let (Some(dst), Some(src)) = (buf.get_mut(..len), bytes.get(..len)) {
            dst.copy_from_slice(src);
        }
        Self { buf, len: len as u8 }
    }

    /// The bytes to write.
    pub fn as_bytes(&self) -> &[u8] {
        self.buf.get(..self.len as usize).unwrap_or(&[])
    }
}

impl Command {
    /// Opcode (first byte).
    pub const fn opcode(&self) -> u8 {
        match self {
            Self::GetFirmwareVersion => 0x01,
            Self::GetBattery => 0x02,
            Self::StreamControl { .. } => 0x03,
            Self::GetSerialNumber => 0x06,
            Self::TimeSync(_) => 0x07,
            Self::GetDeviceName => 0x09,
            Self::GetDeviceTime => 0x0b,
            Self::GetDeviceId => 0x0c,
            Self::OnConnected | Self::StartSession | Self::EndSession => 0x0e,
            Self::HeartRatePeriodic(_) => 0x10,
            Self::SetPromptVolume(_) => 0x36,
            Self::GetImuCalibration => 0xf8,
        }
    }

    /// Wire encoding.
    pub fn encode(&self) -> Encoded {
        let op = self.opcode();
        match *self {
            Self::TimeSync(t) => {
                let b = t.to_be_bytes();
                Encoded::new(&[op, b[0], b[1], b[2], b[3]])
            }
            Self::HeartRatePeriodic(on) => Encoded::new(&[op, u8::from(on)]),
            Self::OnConnected => Encoded::new(&[op, 0x00]),
            Self::StartSession => Encoded::new(&[op, 0x02]),
            Self::EndSession => Encoded::new(&[op, 0x03]),
            Self::StreamControl { mask, enable } => Encoded::new(&[op, mask, u8::from(enable)]),
            Self::SetPromptVolume(v) => Encoded::new(&[op, v]),
            _ => Encoded::new(&[op]),
        }
    }
}

/// A command response: `[opcode, status, payload…]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// Opcode of the command answered.
    pub opcode: u8,
    /// Status byte; `0` = success.
    pub status: u8,
    payload: [u8; Self::MAX_PAYLOAD],
    len: u8,
}

/// Typed view of a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseBody<'a> {
    /// Acknowledgement with no payload.
    Ack,
    /// Firmware version text.
    FirmwareVersion(&'a str),
    /// Battery percentage.
    Battery(u8),
    /// Serial number text.
    SerialNumber(&'a str),
    /// Device name text.
    DeviceName(&'a str),
    /// Device clock (Unix seconds after a time sync).
    DeviceTime(u32),
    /// Device identifier text (the Bluetooth address).
    DeviceId(&'a str),
    /// IMU calibration report: three per-axis `std` bytes and three per-axis `diff` words. Their
    /// exact semantics (and the signedness of `diff`) are not yet verified.
    ImuCalibration {
        /// Per-axis noise estimate.
        std: [u8; 3],
        /// Per-axis offset estimate, raw.
        diff: [u16; 3],
    },
    /// A response this crate does not interpret.
    Other(&'a [u8]),
}

impl Response {
    /// Longest payload accepted (17 bytes observed).
    pub const MAX_PAYLOAD: usize = 64;

    /// Decodes a response.
    pub fn decode(p: &[u8]) -> Result<Self, ParseError> {
        let (&opcode, rest) = p.split_first().ok_or(ParseError::EmptyResponse)?;
        let (status, body) = match rest.split_first() {
            Some((&s, b)) => (s, b),
            None => return Err(ParseError::EmptyResponse),
        };
        if body.len() > Self::MAX_PAYLOAD {
            return Err(ParseError::ResponseLength {
                opcode,
                actual: body.len(),
            });
        }
        let mut payload = [0u8; Self::MAX_PAYLOAD];
        payload
            .get_mut(..body.len())
            .ok_or(ParseError::EmptyResponse)?
            .copy_from_slice(body);
        Ok(Self {
            opcode,
            status,
            payload,
            len: body.len() as u8,
        })
    }

    /// Raw payload after opcode and status.
    pub fn payload(&self) -> &[u8] {
        self.payload.get(..self.len as usize).unwrap_or(&[])
    }

    /// True when the status byte reports success.
    pub fn ok(&self) -> bool {
        self.status == 0
    }

    /// Typed interpretation of the payload.
    pub fn body(&self) -> Result<ResponseBody<'_>, ParseError> {
        let p = self.payload();
        fn as_text<'a>(p: &'a [u8], make: fn(&'a str) -> ResponseBody<'a>) -> ResponseBody<'a> {
            core::str::from_utf8(p).map_or(ResponseBody::Other(p), make)
        }
        let text = |make| as_text(p, make);
        let len_err = ParseError::ResponseLength {
            opcode: self.opcode,
            actual: p.len(),
        };
        Ok(match self.opcode {
            0x01 => text(ResponseBody::FirmwareVersion),
            0x02 => ResponseBody::Battery(*p.first().ok_or(len_err)?),
            0x06 => text(ResponseBody::SerialNumber),
            0x09 => text(ResponseBody::DeviceName),
            0x0b => ResponseBody::DeviceTime(be_u32(p, 0).ok_or(len_err)?),
            0x0c => text(ResponseBody::DeviceId),
            0xf8 => {
                if p.len() != 9 {
                    return Err(len_err);
                }
                ResponseBody::ImuCalibration {
                    std: [
                        *p.first().ok_or(len_err)?,
                        *p.get(1).ok_or(len_err)?,
                        *p.get(2).ok_or(len_err)?,
                    ],
                    diff: [
                        be_u16(p, 3).ok_or(len_err)?,
                        be_u16(p, 5).ok_or(len_err)?,
                        be_u16(p, 7).ok_or(len_err)?,
                    ],
                }
            }
            _ if p.is_empty() => ResponseBody::Ack,
            _ => ResponseBody::Other(p),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings_match_observed_wire_bytes() {
        let cases: [(Command, &[u8]); 10] = [
            (Command::GetFirmwareVersion, &[0x01]),
            (Command::TimeSync(0x6b00_92f6), &[0x07, 0x6b, 0x00, 0x92, 0xf6]),
            (Command::SetPromptVolume(0), &[0x36, 0x00]),
            (Command::HeartRatePeriodic(true), &[0x10, 0x01]),
            (Command::OnConnected, &[0x0e, 0x00]),
            (
                Command::StreamControl {
                    mask: 0xff,
                    enable: true,
                },
                &[0x03, 0xff, 0x01],
            ),
            (
                Command::StreamControl {
                    mask: 0xff,
                    enable: false,
                },
                &[0x03, 0xff, 0x00],
            ),
            (Command::GetImuCalibration, &[0xf8]),
            (Command::StartSession, &[0x0e, 0x02]),
            (Command::EndSession, &[0x0e, 0x03]),
        ];
        for (cmd, bytes) in cases {
            assert_eq!(cmd.encode().as_bytes(), bytes, "{cmd:?}");
        }
    }

    #[test]
    fn responses_decode_like_the_hardware_sent_them() {
        let fw = Response::decode(b"\x01\x00X8.2.2_1.1.19").unwrap();
        assert_eq!(fw.body().unwrap(), ResponseBody::FirmwareVersion("X8.2.2_1.1.19"));
        assert_eq!(
            Response::decode(&[0x02, 0x00, 0x5a]).unwrap().body().unwrap(),
            ResponseBody::Battery(90)
        );
        assert_eq!(
            Response::decode(&[0x0b, 0x00, 0x6b, 0x00, 0x99, 0x70])
                .unwrap()
                .body()
                .unwrap(),
            ResponseBody::DeviceTime(0x6b00_9970)
        );
        let cal = Response::decode(&[0xf8, 0x00, 0x29, 0x2b, 0x28, 0x0d, 0x9a, 0x0e, 0xf4, 0x0b, 0x25]).unwrap();
        assert_eq!(
            cal.body().unwrap(),
            ResponseBody::ImuCalibration {
                std: [41, 43, 40],
                diff: [3482, 3828, 2853]
            }
        );
        let ack = Response::decode(&[0x07, 0x00]).unwrap();
        assert!(ack.ok());
        assert_eq!(ack.body().unwrap(), ResponseBody::Ack);
        assert_eq!(Response::decode(&[0x01]), Err(ParseError::EmptyResponse));
        assert!(Response::decode(&[0xf8, 0x00, 1, 2]).unwrap().body().is_err());
    }
}
