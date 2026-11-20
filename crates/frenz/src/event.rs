//! Events, raw records, and stream health.

use std::collections::BTreeMap;
use std::sync::Arc;

use frenz_proto::stream::StatusKind;
use frenz_proto::{Characteristic, Packet, ParseError};

/// Something recorded verbatim, in arrival order. This is what recorders persist.
#[derive(Debug, Clone)]
pub enum RawRecord {
    /// A notification exactly as received.
    Notification {
        /// Host monotonic time of arrival (ns, see [`crate::clock`]).
        mono_ns: i64,
        /// Host wall time of arrival (Unix ns).
        wall_ns: i64,
        /// Characteristic it arrived on.
        characteristic: Characteristic,
        /// Payload bytes.
        bytes: Arc<[u8]>,
    },
    /// A command written to the band.
    Write {
        /// Host monotonic time of the write (ns).
        mono_ns: i64,
        /// Host wall time (Unix ns).
        wall_ns: i64,
        /// Human-readable label.
        label: String,
        /// Bytes written.
        bytes: Arc<[u8]>,
        /// Write error, if the write failed.
        error: Option<String>,
    },
    /// An annotation (session boundaries, experiment events, …).
    Marker {
        /// Host monotonic time (ns).
        mono_ns: i64,
        /// Host wall time (Unix ns).
        wall_ns: i64,
        /// Label.
        label: String,
    },
}

/// A decoded event delivered to subscribers.
#[derive(Debug, Clone)]
pub enum Event {
    /// A decoded packet with its host arrival time.
    Packet {
        /// Host monotonic arrival time (ns).
        mono_ns: i64,
        /// The packet.
        packet: Arc<Packet>,
    },
    /// A notification that could not be decoded. It is still in the raw record.
    DecodeError {
        /// Host monotonic arrival time (ns).
        mono_ns: i64,
        /// Characteristic it arrived on.
        characteristic: Characteristic,
        /// Why.
        error: ParseError,
    },
    /// EEG frames are missing: the sequence counter jumped.
    SequenceGap {
        /// Host monotonic arrival time of the frame after the gap (ns).
        mono_ns: i64,
        /// Sequence number expected.
        expected: u8,
        /// Sequence number received.
        received: u8,
    },
    /// The EEG sequence counter restarted at 0 right after a stream-start command (not data loss).
    CounterRestart {
        /// Host monotonic arrival time (ns).
        mono_ns: i64,
    },
    /// The band reports no electrode contact; it powers itself off about 33 s later.
    OffHead {
        /// Host monotonic arrival time (ns).
        mono_ns: i64,
    },
    /// Contact restored after [`Event::OffHead`].
    OnHead {
        /// Host monotonic arrival time (ns).
        mono_ns: i64,
    },
    /// Status message seen seconds before the band powered itself off (inferred meaning).
    PowerOffWarning {
        /// Host monotonic arrival time (ns).
        mono_ns: i64,
    },
    /// The Bluetooth link was lost or closed.
    Disconnected {
        /// Host monotonic time (ns).
        mono_ns: i64,
    },
    /// A reconnection attempt is starting.
    Reconnecting {
        /// Host monotonic time (ns).
        mono_ns: i64,
        /// Attempt number since the link was lost, from 1.
        attempt: u32,
    },
    /// The link is back and the start sequence has been sent again. Data after this belongs to a
    /// new segment.
    Reconnected {
        /// Host monotonic time (ns).
        mono_ns: i64,
    },
    /// Reconnection was abandoned after the configured time.
    ReconnectFailed {
        /// Host monotonic time (ns).
        mono_ns: i64,
    },
}

/// Running counters for a connection. Every irregularity increments a counter here.
#[derive(Debug, Clone, Default)]
pub struct Health {
    /// Notifications received per characteristic name.
    pub notifications: BTreeMap<&'static str, u64>,
    /// Notifications that failed to decode, per characteristic name.
    pub decode_errors: BTreeMap<&'static str, u64>,
    /// EEG frames without a sequence byte or with an unknown frame type.
    pub irregular_eeg_frames: u64,
    /// EEG sequence gaps (lost frames, or restarts not explained by a start command).
    pub sequence_gaps: u64,
    /// Estimated EEG frames lost in those gaps.
    pub frames_lost: u64,
    /// Counter restarts following a stream-start command.
    pub counter_restarts: u64,
    /// Last battery percentage reported.
    pub battery_percent: Option<u8>,
    /// Last contact state reported.
    pub contact_state: Option<u8>,
    /// Times the link was re-established after being lost.
    pub reconnects: u64,
    /// Whether the link is up.
    pub connected: bool,
}

/// Time after a stream start during which a zero contact state is not reported as off-head.
const CONTACT_SETTLE_NS: i64 = 6_000_000_000;

/// Turns raw notifications into events and keeps [`Health`]. Used live by [`crate::Band`] and
/// offline to check recordings.
#[derive(Debug, Default)]
pub struct Decoder {
    eeg_sequence: Option<u8>,
    off_head: bool,
}

impl Decoder {
    /// Forgets the last EEG sequence number (after the link was re-established).
    pub fn reset(&mut self) {
        self.eeg_sequence = None;
    }

    /// Decodes one notification. `last_start_ns` is the time of the most recent stream-start write.
    pub fn handle(
        &mut self,
        mono_ns: i64,
        characteristic: Characteristic,
        bytes: &[u8],
        last_start_ns: i64,
        health: &mut Health,
        out: &mut Vec<Event>,
    ) {
        *health.notifications.entry(characteristic.name()).or_default() += 1;
        let packet = match Packet::decode(characteristic, bytes) {
            Ok(p) => p,
            Err(error) => {
                if matches!(error, ParseError::EegFrameType(_) | ParseError::EegUnsequenced) {
                    health.irregular_eeg_frames += 1;
                    self.eeg_sequence = None;
                } else {
                    *health.decode_errors.entry(characteristic.name()).or_default() += 1;
                }
                out.push(Event::DecodeError {
                    mono_ns,
                    characteristic,
                    error,
                });
                return;
            }
        };
        match &packet {
            Packet::Eeg(e) => {
                if let Some(prev) = self.eeg_sequence {
                    let expected = prev.wrapping_add(1);
                    if e.sequence != expected {
                        let after_start = (0..1_000_000_000).contains(&(mono_ns - last_start_ns));
                        if e.sequence == 0 && after_start {
                            health.counter_restarts += 1;
                            out.push(Event::CounterRestart { mono_ns });
                        } else {
                            health.sequence_gaps += 1;
                            health.frames_lost += u64::from(e.sequence.wrapping_sub(expected));
                            out.push(Event::SequenceGap {
                                mono_ns,
                                expected,
                                received: e.sequence,
                            });
                        }
                    }
                }
                self.eeg_sequence = Some(e.sequence);
            }
            Packet::Battery(b) => health.battery_percent = Some(b.percent),
            Packet::Contact(c) => {
                health.contact_state = Some(c.state);
                // After a stream start the state reads 0, 0, 1, 1, 3 over ~10 s while it settles
                // (observed in every session), so an early 0 does not mean the band is off.
                let settling = (0..CONTACT_SETTLE_NS).contains(&(mono_ns - last_start_ns));
                if c.off_head() && !self.off_head && !settling {
                    self.off_head = true;
                    out.push(Event::OffHead { mono_ns });
                } else if !c.off_head() && self.off_head {
                    self.off_head = false;
                    out.push(Event::OnHead { mono_ns });
                }
            }
            Packet::Status(s) if s.kind() == StatusKind::PowerOffWarning => {
                out.push(Event::PowerOffWarning { mono_ns });
            }
            _ => {}
        }
        out.push(Event::Packet {
            mono_ns,
            packet: Arc::new(packet),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eeg(seq: u8) -> Vec<u8> {
        let mut p = vec![0u8; 178];
        p[0] = seq;
        p[1] = 6;
        p
    }

    #[test]
    fn classifies_gaps_and_restarts() {
        let (mut d, mut h, mut out) = (Decoder::default(), Health::default(), Vec::new());
        for (t, seq) in [(0, 254u8), (10, 255), (20, 0), (30, 3)] {
            d.handle(t, Characteristic::Eeg, &eeg(seq), -10_000_000_000, &mut h, &mut out);
        }
        assert_eq!((h.sequence_gaps, h.frames_lost), (1, 2));
        // A restart at 0 within 1 s of a stream-start write is not a gap.
        d.handle(2_000, Characteristic::Eeg, &eeg(0), 1_500, &mut h, &mut out);
        assert_eq!((h.sequence_gaps, h.counter_restarts), (1, 1));
        let mut shifted = eeg(0);
        shifted[0] = 6;
        shifted[1] = 0x6b;
        d.handle(3_000, Characteristic::Eeg, &shifted, 0, &mut h, &mut out);
        assert_eq!(h.irregular_eeg_frames, 1);
        assert!(h.decode_errors.is_empty());
    }

    #[test]
    fn contact_transitions() {
        let (mut d, mut h, mut out) = (Decoder::default(), Health::default(), Vec::new());
        // Zeros while settling after a stream start are not off-head.
        for (t, state) in [(2_000_000_000, 0u8), (4_000_000_000, 0)] {
            d.handle(t, Characteristic::Contact, &[0, 0, 0, 1, state], 0, &mut h, &mut out);
        }
        assert!(out.iter().all(|e| !matches!(e, Event::OffHead { .. })));
        for state in [3u8, 0, 0, 1] {
            d.handle(
                60_000_000_000,
                Characteristic::Contact,
                &[0, 0, 0, 1, state],
                0,
                &mut h,
                &mut out,
            );
        }
        let kinds: Vec<_> = out
            .iter()
            .filter(|e| matches!(e, Event::OffHead { .. } | Event::OnHead { .. }))
            .map(|e| matches!(e, Event::OffHead { .. }))
            .collect();
        assert_eq!(kinds, [true, false]);
    }
}
