//! GATT identifiers of the band's own services.
//!
//! The band also exposes Google Fast Pair, audio-SoC and standard battery/device-information
//! services. Clients must not subscribe to those: doing so dropped the connection on hardware.

/// UUID of a 128-bit characteristic `4542xxxx-0000-ffff-ff45-415241424c45`.
const fn frenz_uuid(short: u32) -> u128 {
    ((short as u128) << 96) | 0x0000_0000_0000_ffff_ff45_4152_4142_4c45
}

/// Command service.
pub const COMMAND_SERVICE: u128 = frenz_uuid(0x4542_0100);
/// Stream service.
pub const STREAM_SERVICE: u128 = frenz_uuid(0x4542_0200);

/// A characteristic of the band's own services.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Characteristic {
    /// Command input (write without response).
    CommandRx,
    /// Command responses `[opcode, status, payload…]`.
    CommandResponse,
    /// Status/event messages (heartbeats, periodic status blocks, shutdown warning).
    Status,
    /// EEG, 125 Hz.
    Eeg,
    /// Accelerometer, 50 Hz.
    Accelerometer,
    /// Photoplethysmography, ~25 Hz.
    Ppg,
    /// Heart rate.
    HeartRate,
    /// Blood-oxygen saturation.
    Spo2,
    /// Battery percentage.
    Battery,
    /// Gyroscope, 50 Hz.
    Gyroscope,
    /// Electrode contact state.
    Contact,
}

impl Characteristic {
    /// Every characteristic, in GATT order.
    pub const ALL: [Self; 11] = [
        Self::CommandRx,
        Self::CommandResponse,
        Self::Status,
        Self::Eeg,
        Self::Accelerometer,
        Self::Ppg,
        Self::HeartRate,
        Self::Spo2,
        Self::Battery,
        Self::Gyroscope,
        Self::Contact,
    ];

    /// Characteristics a client subscribes to (all notify characteristics of the band's services).
    pub const NOTIFY: [Self; 10] = [
        Self::CommandResponse,
        Self::Status,
        Self::Eeg,
        Self::Accelerometer,
        Self::Ppg,
        Self::HeartRate,
        Self::Spo2,
        Self::Battery,
        Self::Gyroscope,
        Self::Contact,
    ];

    /// The 128-bit UUID.
    pub const fn uuid(self) -> u128 {
        frenz_uuid(match self {
            Self::CommandRx => 0x4542_0101,
            Self::CommandResponse => 0x4542_0102,
            Self::Status => 0x4542_0103,
            Self::Eeg => 0x4542_0201,
            Self::Accelerometer => 0x4542_0202,
            Self::Ppg => 0x4542_0203,
            Self::HeartRate => 0x4542_0204,
            Self::Spo2 => 0x4542_0205,
            Self::Battery => 0x4542_0206,
            Self::Gyroscope => 0x4542_0207,
            Self::Contact => 0x4542_0208,
        })
    }

    /// Looks up a characteristic by UUID.
    pub fn from_uuid(uuid: u128) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.uuid() == uuid)
    }

    /// Short stable name, used in recordings and logs.
    pub const fn name(self) -> &'static str {
        match self {
            Self::CommandRx => "command_rx",
            Self::CommandResponse => "command_response",
            Self::Status => "status",
            Self::Eeg => "eeg",
            Self::Accelerometer => "accelerometer",
            Self::Ppg => "ppg",
            Self::HeartRate => "heart_rate",
            Self::Spo2 => "spo2",
            Self::Battery => "battery",
            Self::Gyroscope => "gyroscope",
            Self::Contact => "contact",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuids_match_the_wire() {
        assert_eq!(Characteristic::Eeg.uuid(), 0x4542_0201_0000_ffff_ff45_4152_4142_4c45);
        assert_eq!(COMMAND_SERVICE, 0x4542_0100_0000_ffff_ff45_4152_4142_4c45);
        for c in Characteristic::ALL {
            assert_eq!(Characteristic::from_uuid(c.uuid()), Some(c));
        }
        assert_eq!(Characteristic::from_uuid(0), None);
    }
}
