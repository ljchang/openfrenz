//! Decoders for notification payloads.
//!
//! Device times are the band's whole-second clock: seconds since power-on until the host sends
//! [`Command::TimeSync`](crate::Command::TimeSync), Unix seconds afterwards.

use crate::command::Response;
use crate::error::ParseError;
use crate::gatt::Characteristic;
use crate::{be_u16, be_u32};

/// Any decoded notification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Packet {
    /// EEG frame (9 samples).
    Eeg(EegPacket),
    /// Accelerometer frame (25 samples).
    Accelerometer(MotionPacket),
    /// Gyroscope frame (25 samples).
    Gyroscope(MotionPacket),
    /// PPG frame (12–13 samples).
    Ppg(PpgPacket),
    /// Heart-rate reading.
    HeartRate(VitalPacket),
    /// SpO2 reading.
    Spo2(VitalPacket),
    /// Battery percentage.
    Battery(BatteryPacket),
    /// Electrode contact state.
    Contact(ContactPacket),
    /// Message on the status channel.
    Status(StatusMessage),
    /// Command response.
    Response(Response),
}

impl Packet {
    /// Decodes a notification received on `characteristic`.
    pub fn decode(characteristic: Characteristic, payload: &[u8]) -> Result<Self, ParseError> {
        Ok(match characteristic {
            Characteristic::Eeg => Self::Eeg(EegPacket::decode(payload)?),
            Characteristic::Accelerometer => Self::Accelerometer(MotionPacket::decode(characteristic, payload)?),
            Characteristic::Gyroscope => Self::Gyroscope(MotionPacket::decode(characteristic, payload)?),
            Characteristic::Ppg => Self::Ppg(PpgPacket::decode(payload)?),
            Characteristic::HeartRate => Self::HeartRate(VitalPacket::decode(characteristic, payload)?),
            Characteristic::Spo2 => Self::Spo2(VitalPacket::decode(characteristic, payload)?),
            Characteristic::Battery => Self::Battery(BatteryPacket::decode(payload)?),
            Characteristic::Contact => Self::Contact(ContactPacket::decode(payload)?),
            Characteristic::Status => Self::Status(StatusMessage::decode(payload)?),
            Characteristic::CommandResponse => Self::Response(Response::decode(payload)?),
            Characteristic::CommandRx => return Err(ParseError::UnknownCharacteristic),
        })
    }

    /// The device time carried by the packet, if it has one.
    pub fn device_time(&self) -> Option<u32> {
        match self {
            Self::Eeg(p) => Some(p.device_time),
            Self::Accelerometer(p) | Self::Gyroscope(p) => Some(p.device_time),
            Self::Ppg(p) => Some(p.device_time),
            Self::HeartRate(p) | Self::Spo2(p) => Some(p.device_time),
            Self::Battery(p) => Some(p.device_time),
            Self::Contact(p) => Some(p.device_time),
            Self::Status(_) | Self::Response(_) => None,
        }
    }
}

/// Sign-extends a 24-bit big-endian integer.
fn i24(b: [u8; 3]) -> i32 {
    i32::from_be_bytes([b[0], b[1], b[2], 0]) >> 8
}

fn read3(bytes: &[u8], at: usize) -> Option<[u8; 3]> {
    Some([*bytes.get(at)?, *bytes.get(at + 1)?, *bytes.get(at + 2)?])
}

/// EEG frame: 9 samples × 6 channel slots of signed 24-bit ADC counts.
///
/// Only slots 0, 1, 3 and 4 carry signal on the verified hardware ([`EegPacket::ACTIVE_CHANNELS`]);
/// slots 2 and 5 are always zero. Units are raw ADC counts; the microvolt scale is not yet known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EegPacket {
    /// 8-bit frame counter; consecutive frames differ by 1 (mod 256).
    pub sequence: u8,
    /// Device time (whole seconds).
    pub device_time: u32,
    /// `samples[i][slot]`, raw signed 24-bit counts.
    pub samples: [[i32; 6]; 9],
    /// Per-sample flag byte (meaning unknown; 1 on ~4 % of samples).
    pub flags: [u8; 9],
    /// Final byte of the frame (meaning unknown; observed 0–16).
    pub trailer: u8,
}

impl EegPacket {
    /// Verified frame length.
    pub const LEN: usize = 178;
    /// Frame type byte of EEG frames (equals the channel-slot count).
    pub const FRAME_TYPE: u8 = 6;
    /// Samples per frame.
    pub const SAMPLES: usize = 9;
    /// Nominal sample rate.
    pub const RATE_HZ: f64 = 125.0;
    /// Channel slots that carry signal, and the electrode each corresponds to (press test).
    pub const ACTIVE_CHANNELS: [(usize, &'static str); 4] = [
        (0, "forehead_right"),
        (1, "comb_left"),
        (3, "forehead_left"),
        (4, "comb_right"),
    ];

    /// Decodes a frame.
    pub fn decode(p: &[u8]) -> Result<Self, ParseError> {
        let frame_type = *p.get(1).ok_or(ParseError::Length {
            characteristic: Characteristic::Eeg,
            actual: p.len(),
        })?;
        if frame_type != Self::FRAME_TYPE {
            if p.len() == Self::LEN && p.first() == Some(&Self::FRAME_TYPE) {
                return Err(ParseError::EegUnsequenced);
            }
            return Err(ParseError::EegFrameType(frame_type));
        }
        if p.len() != Self::LEN {
            return Err(ParseError::Length {
                characteristic: Characteristic::Eeg,
                actual: p.len(),
            });
        }
        let bad = ParseError::Length {
            characteristic: Characteristic::Eeg,
            actual: p.len(),
        };
        let mut samples = [[0i32; 6]; 9];
        let mut flags = [0u8; 9];
        for (i, (row, flag)) in samples.iter_mut().zip(flags.iter_mut()).enumerate() {
            let base = 6 + 19 * i;
            for (j, v) in row.iter_mut().enumerate() {
                *v = i24(read3(p, base + 3 * j).ok_or(bad)?);
            }
            *flag = *p.get(base + 18).ok_or(bad)?;
        }
        Ok(Self {
            sequence: *p.first().ok_or(bad)?,
            device_time: be_u32(p, 2).ok_or(bad)?,
            samples,
            flags,
            trailer: *p.get(Self::LEN - 1).ok_or(bad)?,
        })
    }
}

/// Accelerometer or gyroscope frame: 25 samples of x, y, z signed 16-bit counts.
///
/// Sensor frame (verified): +x down (toward the chin when worn), +y toward the wearer's left,
/// +z forward. Accelerometer ≈ 16 384 counts/g (±2 g); gyroscope scale unknown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionPacket {
    /// Device time (whole seconds).
    pub device_time: u32,
    /// `samples[i] = [x, y, z]` raw counts.
    pub samples: [[i16; 3]; 25],
    /// Last four bytes (meaning unknown; mostly zero).
    pub tail: [u8; 4],
}

impl MotionPacket {
    /// Verified frame length.
    pub const LEN: usize = 158;
    /// Samples per frame.
    pub const SAMPLES: usize = 25;
    /// Nominal sample rate.
    pub const RATE_HZ: f64 = 50.0;

    /// Decodes a frame received on `characteristic` (used only for error reporting).
    pub fn decode(characteristic: Characteristic, p: &[u8]) -> Result<Self, ParseError> {
        let bad = ParseError::Length {
            characteristic,
            actual: p.len(),
        };
        if p.len() != Self::LEN {
            return Err(bad);
        }
        let mut samples = [[0i16; 3]; 25];
        for (i, s) in samples.iter_mut().enumerate() {
            for (a, v) in s.iter_mut().enumerate() {
                *v = be_u16(p, 4 + 6 * i + 2 * a).ok_or(bad)? as i16;
            }
        }
        let t = p.get(154..158).ok_or(bad)?;
        Ok(Self {
            device_time: be_u32(p, 0).ok_or(bad)?,
            samples,
            tail: [
                *t.first().ok_or(bad)?,
                *t.get(1).ok_or(bad)?,
                *t.get(2).ok_or(bad)?,
                *t.get(3).ok_or(bad)?,
            ],
        })
    }
}

/// PPG frame: up to [`PpgPacket::MAX_SAMPLES`] samples of three unsigned 19-bit optical channels.
///
/// Channel `k` holds the word tagged `k + 1`. Channel 0 (tag 1) is always zero on the verified
/// hardware; which LED each tag drives is not yet known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PpgPacket {
    /// Device time (whole seconds).
    pub device_time: u32,
    samples: [[u32; 3]; Self::MAX_SAMPLES],
    len: u8,
    /// Final two bytes (always zero so far).
    pub trailer: [u8; 2],
}

impl PpgPacket {
    /// Largest sample count accepted (12 and 13 observed).
    pub const MAX_SAMPLES: usize = 16;
    /// Nominal sample rate.
    pub const RATE_HZ: f64 = 25.0;

    /// Decoded samples, `[channel0, channel1, channel2]` each.
    pub fn samples(&self) -> &[[u32; 3]] {
        self.samples.get(..self.len as usize).unwrap_or(&[])
    }

    /// Decodes a frame.
    pub fn decode(p: &[u8]) -> Result<Self, ParseError> {
        let bad = ParseError::Length {
            characteristic: Characteristic::Ppg,
            actual: p.len(),
        };
        let body = p.len().checked_sub(6).ok_or(bad)?;
        let n = body / 9;
        if body % 9 != 0 || n == 0 || n > Self::MAX_SAMPLES {
            return Err(bad);
        }
        let mut samples = [[0u32; 3]; Self::MAX_SAMPLES];
        for (i, s) in samples.iter_mut().take(n).enumerate() {
            let mut seen = [false; 3];
            for k in 0..3 {
                let w = read3(p, 4 + 9 * i + 3 * k).ok_or(bad)?;
                let tag = usize::from(w[0] >> 3);
                let value = (u32::from(w[0] & 0x07) << 16) | (u32::from(w[1]) << 8) | u32::from(w[2]);
                let col = tag
                    .checked_sub(1)
                    .filter(|c| *c < 3)
                    .ok_or(ParseError::PpgTags { sample: i })?;
                let slot = seen.get_mut(col).ok_or(ParseError::PpgTags { sample: i })?;
                if *slot {
                    return Err(ParseError::PpgTags { sample: i });
                }
                *slot = true;
                *s.get_mut(col).ok_or(ParseError::PpgTags { sample: i })? = value;
            }
        }
        Ok(Self {
            device_time: be_u32(p, 0).ok_or(bad)?,
            samples,
            len: n as u8,
            trailer: [*p.get(p.len() - 2).ok_or(bad)?, *p.last().ok_or(bad)?],
        })
    }
}

/// Heart-rate or SpO2 reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VitalPacket {
    /// Device time (whole seconds).
    pub device_time: u32,
    /// Byte 4 (meaning unknown).
    pub check: u8,
    /// Reading (beats per minute or percent saturation).
    pub value: u8,
}

impl VitalPacket {
    /// Verified length.
    pub const LEN: usize = 6;

    /// Decodes a reading received on `characteristic` (used only for error reporting).
    pub fn decode(characteristic: Characteristic, p: &[u8]) -> Result<Self, ParseError> {
        let bad = ParseError::Length {
            characteristic,
            actual: p.len(),
        };
        if p.len() != Self::LEN {
            return Err(bad);
        }
        Ok(Self {
            device_time: be_u32(p, 0).ok_or(bad)?,
            check: *p.get(4).ok_or(bad)?,
            value: *p.get(5).ok_or(bad)?,
        })
    }
}

fn time_and_byte(characteristic: Characteristic, p: &[u8]) -> Result<(u32, u8), ParseError> {
    let bad = ParseError::Length {
        characteristic,
        actual: p.len(),
    };
    if p.len() != 5 {
        return Err(bad);
    }
    Ok((be_u32(p, 0).ok_or(bad)?, *p.get(4).ok_or(bad)?))
}

/// Battery level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatteryPacket {
    /// Device time (whole seconds).
    pub device_time: u32,
    /// Charge in percent (matches the standard GATT battery level).
    pub percent: u8,
}

impl BatteryPacket {
    /// Decodes a battery notification.
    pub fn decode(p: &[u8]) -> Result<Self, ParseError> {
        let (device_time, percent) = time_and_byte(Characteristic::Battery, p)?;
        Ok(Self { device_time, percent })
    }
}

/// Electrode contact state, sent every 2 s.
///
/// Observed: 3 while worn, 2 intermittently while worn, 1 shortly after putting the band on,
/// 0 once removed, and 0, 0, 1, 1, 3 over ~10 s after every stream start. Probably a 0–3 contact-quality
/// level (unverified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContactPacket {
    /// Device time (whole seconds).
    pub device_time: u32,
    /// Raw state byte.
    pub state: u8,
}

impl ContactPacket {
    /// Decodes a contact notification.
    pub fn decode(p: &[u8]) -> Result<Self, ParseError> {
        let (device_time, state) = time_and_byte(Characteristic::Contact, p)?;
        Ok(Self { device_time, state })
    }

    /// True when the band reports no contact at all (it powers off ~33 s later).
    pub fn off_head(&self) -> bool {
        self.state == 0
    }
}

/// A message on the status channel, kept verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusMessage {
    bytes: [u8; Self::MAX_LEN],
    len: u8,
}

/// Interpretation of a status message. Only [`StatusKind::Heartbeat`] is a structural certainty;
/// the others are inferred from timing and may be renamed as they are verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusKind {
    /// `0e 00 00 00 00`, about once a second.
    Heartbeat,
    /// `07 db` or `07 d0`: seen seconds before the band powered itself off when off-head.
    PowerOffWarning,
    /// `07 11`, `07 12 …`: emitted as a block every 60 s.
    Periodic,
    /// Anything else.
    Other,
}

impl StatusMessage {
    /// Longest message accepted (6 bytes observed).
    pub const MAX_LEN: usize = 32;

    /// Keeps a status message verbatim.
    pub fn decode(p: &[u8]) -> Result<Self, ParseError> {
        if p.is_empty() || p.len() > Self::MAX_LEN {
            return Err(ParseError::Length {
                characteristic: Characteristic::Status,
                actual: p.len(),
            });
        }
        let mut bytes = [0u8; Self::MAX_LEN];
        bytes
            .get_mut(..p.len())
            .ok_or(ParseError::EmptyResponse)?
            .copy_from_slice(p);
        Ok(Self {
            bytes,
            len: p.len() as u8,
        })
    }

    /// The raw message.
    pub fn bytes(&self) -> &[u8] {
        self.bytes.get(..self.len as usize).unwrap_or(&[])
    }

    /// Best-known interpretation.
    pub fn kind(&self) -> StatusKind {
        match self.bytes() {
            [0x0e, 0, 0, 0, 0] => StatusKind::Heartbeat,
            [0x07, 0xdb] | [0x07, 0xd0] => StatusKind::PowerOffWarning,
            [0x07, 0x11] | [0x07, 0x12, ..] => StatusKind::Periodic,
            _ => StatusKind::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eeg_frame(seq: u8, secs: u32, value: impl Fn(usize, usize) -> i32) -> [u8; 178] {
        let mut p = [0u8; 178];
        p[0] = seq;
        p[1] = 6;
        p[2..6].copy_from_slice(&secs.to_be_bytes());
        for i in 0..9 {
            for j in 0..6 {
                let v = value(i, j).to_be_bytes();
                p[6 + 19 * i + 3 * j..6 + 19 * i + 3 * j + 3].copy_from_slice(&v[1..4]);
            }
            p[6 + 19 * i + 18] = (i % 2) as u8;
        }
        p[177] = 9;
        p
    }

    #[test]
    fn eeg_decodes_signed_24_bit_samples() {
        let p = eeg_frame(200, 1_795_201_441, |i, j| {
            (i as i32 * 100 + j as i32) * if j % 2 == 0 { -1 } else { 1 }
        });
        let e = EegPacket::decode(&p).unwrap();
        assert_eq!((e.sequence, e.device_time, e.trailer), (200, 1_795_201_441, 9));
        assert_eq!(e.samples[3], [-300, 301, -302, 303, -304, 305]);
        assert_eq!(e.flags[..3], [0, 1, 0]);
        let extremes = eeg_frame(0, 0, |_, j| if j == 0 { -(1 << 23) } else { (1 << 23) - 1 });
        assert_eq!(
            EegPacket::decode(&extremes).unwrap().samples[0][..2],
            [-(1 << 23), (1 << 23) - 1]
        );
    }

    #[test]
    fn eeg_rejects_other_frame_types_and_lengths() {
        let mut p = eeg_frame(0, 0, |_, _| 0);
        p[1] = 106;
        assert_eq!(EegPacket::decode(&p), Err(ParseError::EegFrameType(106)));
        p[0] = 6; // unsequenced variant: type byte first, Unix time follows
        assert_eq!(EegPacket::decode(&p), Err(ParseError::EegUnsequenced));
        p[1] = 6;
        assert!(matches!(
            EegPacket::decode(&p[..177]),
            Err(ParseError::Length { actual: 177, .. })
        ));
        assert!(EegPacket::decode(&[]).is_err());
    }

    #[test]
    fn motion_decodes_big_endian_i16() {
        let mut p = [0u8; 158];
        p[0..4].copy_from_slice(&383u32.to_be_bytes());
        p[4..6].copy_from_slice(&(-14799i16).to_be_bytes());
        p[152..154].copy_from_slice(&16384i16.to_be_bytes());
        p[154..158].copy_from_slice(&[0x01, 0xff, 0xe0, 0x00]);
        let m = MotionPacket::decode(Characteristic::Accelerometer, &p).unwrap();
        assert_eq!(
            (m.device_time, m.samples[0][0], m.samples[24][2], m.tail),
            (383, -14799, 16384, [1, 0xff, 0xe0, 0])
        );
        assert!(MotionPacket::decode(Characteristic::Accelerometer, &p[..154]).is_err());
    }

    fn ppg_word(tag: u8, value: u32) -> [u8; 3] {
        let w = (u32::from(tag) << 19) | (value & 0x7ffff);
        [(w >> 16) as u8, (w >> 8) as u8, w as u8]
    }

    #[test]
    fn ppg_decodes_by_tag_in_any_order() {
        let mut p = vec![0x6b, 0x00, 0x99, 0xa7];
        for i in 0..13u32 {
            for (tag, v) in [(1, 0), (3, 135_000 + i), (2, 0x7ffff - i)] {
                p.extend_from_slice(&ppg_word(tag, v));
            }
        }
        p.extend_from_slice(&[0, 0]);
        assert_eq!(p.len(), 123);
        let d = PpgPacket::decode(&p).unwrap();
        assert_eq!(d.samples().len(), 13);
        assert_eq!(d.samples()[12], [0, 0x7ffff - 12, 135_012]);
    }

    #[test]
    fn ppg_rejects_duplicate_or_unknown_tags() {
        let mut p = vec![0, 0, 0, 0];
        for tag in [1, 1, 2] {
            p.extend_from_slice(&ppg_word(tag, 5));
        }
        p.extend_from_slice(&[0, 0]);
        assert_eq!(PpgPacket::decode(&p), Err(ParseError::PpgTags { sample: 0 }));
        p[4] = 5 << 3;
        assert_eq!(PpgPacket::decode(&p), Err(ParseError::PpgTags { sample: 0 }));
        assert!(PpgPacket::decode(&p[..10]).is_err());
    }

    #[test]
    fn small_packets() {
        let v = VitalPacket::decode(Characteristic::HeartRate, &[0x6b, 0, 0x92, 0xf6, 1, 72]).unwrap();
        assert_eq!((v.device_time, v.check, v.value), (1_795_199_734, 1, 72));
        assert_eq!(BatteryPacket::decode(&[0, 0, 1, 0x81, 0x5a]).unwrap().percent, 90);
        assert!(ContactPacket::decode(&[0, 0, 1, 0x80, 0]).unwrap().off_head());
        assert_eq!(
            StatusMessage::decode(&[0x0e, 0, 0, 0, 0]).unwrap().kind(),
            StatusKind::Heartbeat
        );
        assert_eq!(
            StatusMessage::decode(&[0x07, 0xdb]).unwrap().kind(),
            StatusKind::PowerOffWarning
        );
        assert_eq!(
            StatusMessage::decode(&[0x07, 0x12, 0xaa, 0xaa, 0, 0x25])
                .unwrap()
                .kind(),
            StatusKind::Periodic
        );
        assert!(StatusMessage::decode(&[]).is_err());
    }
}
