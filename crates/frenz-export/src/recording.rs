//! Sessions from SQLite recordings written by the `frenz` crate.

use std::path::Path;

use frenz::{Decoder, Health};
use frenz_proto::{Characteristic, Packet};
use rusqlite::{Connection, OpenFlags};
use serde_json::{Map, Value, json};

use crate::table::{Column, Table};
use crate::timing::{ClockFit, Segmenter, fit_lower_envelope};
use crate::{Error, Session, eeg_channel_json, motion_json};

/// The EEG sequence counter restarts at 0 within this time of a stream-start command (observed).
const RESTART_WINDOW_NS: i64 = 1_000_000_000;

/// One received packet's samples, before timing.
struct PacketRows<T> {
    arrival: f64,
    mono: i64,
    device_time: u32,
    sequence: Option<u8>,
    samples: Vec<T>,
}

/// Assigns segment and sample indices, fits each segment's clock, and returns per-sample
/// `(t, segment, packet, sample_in_packet, duplicate)` plus the fits.
struct Indexed {
    t: Vec<f64>,
    segment: Vec<i64>,
    packet: Vec<i64>,
    sample: Vec<i8>,
    duplicate: Vec<bool>,
    held: Vec<bool>,
    fits: Vec<ClockFit>,
}

fn index_and_time<T: PartialEq>(
    packets: &[PacketRows<T>],
    break_before: &[bool],
    starts: &[i64],
    rate_hz: f64,
    lost_frames: &mut u64,
    restarts: &mut u64,
) -> Indexed {
    // Per-packet: segment and index of its first sample within the segment.
    let mut seg_of = Vec::with_capacity(packets.len());
    let mut first_sample = Vec::with_capacity(packets.len());
    let mut segmenter = Segmenter::new(rate_hz);
    for (i, p) in packets.iter().enumerate() {
        let forced = break_before.get(i).copied().unwrap_or(false);
        let after_start = starts.iter().any(|s| (0..RESTART_WINDOW_NS).contains(&(p.mono - s)));
        let placed = segmenter.push(
            p.arrival,
            p.device_time,
            p.sequence,
            p.samples.len(),
            forced,
            after_start,
        );
        seg_of.push(placed.segment);
        first_sample.push(placed.first_sample);
    }
    *lost_frames += segmenter.lost_frames;
    *restarts += segmenter.restarts;
    // Fit each segment on non-duplicate packets.
    let segments = seg_of.last().map_or(0, |s| s + 1);
    let mut fits = Vec::new();
    for s in 0..segments {
        let points: Vec<(f64, f64)> = packets
            .iter()
            .enumerate()
            .filter(|(i, p)| seg_of.get(*i) == Some(&s) && !is_duplicate(packets, *i) && !is_held(p))
            .map(|(i, p)| {
                (
                    (first_sample.get(i).copied().unwrap_or(0) + p.samples.len() as i64 - 1) as f64,
                    p.arrival,
                )
            })
            .collect();
        fits.push(fit_lower_envelope(&points, rate_hz));
    }
    let mut out = Indexed {
        t: vec![],
        segment: vec![],
        packet: vec![],
        sample: vec![],
        duplicate: vec![],
        held: vec![],
        fits,
    };
    for (i, p) in packets.iter().enumerate() {
        let s = seg_of.get(i).copied().unwrap_or(0);
        let fit = out.fits.get(s as usize).copied();
        let dup = is_duplicate(packets, i);
        let held = is_held(p);
        for k in 0..p.samples.len() {
            let n = first_sample.get(i).copied().unwrap_or(0) + k as i64;
            out.t.push(fit.map_or(p.arrival, |f| f.time(n as f64)));
            out.segment.push(s);
            out.packet.push(i as i64);
            out.sample.push(k as i8);
            out.duplicate.push(dup);
            out.held.push(held);
        }
    }
    out
}

/// A packet whose samples are all identical: the sensor output is frozen (seen on PPG for seconds
/// at a time while electrode contact fluctuated).
fn is_held<T: PartialEq>(p: &PacketRows<T>) -> bool {
    p.samples.len() > 1 && p.samples.windows(2).all(|w| w.first() == w.get(1))
}

/// A packet whose samples equal the previous packet's (the band has been seen to resend a block).
fn is_duplicate<T: PartialEq>(packets: &[PacketRows<T>], i: usize) -> bool {
    match (i.checked_sub(1).and_then(|j| packets.get(j)), packets.get(i)) {
        (Some(a), Some(b)) => a.samples == b.samples,
        _ => false,
    }
}

fn median(mut v: Vec<i64>) -> i64 {
    v.sort_unstable();
    v.get(v.len() / 2).copied().unwrap_or_default()
}

fn health_json(h: &Health) -> Value {
    json!({
        "notifications": h.notifications, "decode_errors": h.decode_errors,
        "irregular_eeg_frames": h.irregular_eeg_frames, "sequence_gaps": h.sequence_gaps,
        "frames_lost": h.frames_lost, "counter_restarts": h.counter_restarts,
    })
}

/// Reads a recording.
pub fn from_recording(path: &Path) -> Result<Session, Error> {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut recording_meta = Map::new();
    for row in db
        .prepare("SELECT key, value FROM meta")?
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
    {
        let (k, v) = row?;
        recording_meta.insert(k, serde_json::from_str(&v).unwrap_or(Value::String(v)));
    }
    let notifications: Vec<(i64, i64, String, Vec<u8>)> = db
        .prepare("SELECT mono_ns, wall_ns, char, payload FROM notifications ORDER BY id")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<Result<_, _>>()?;
    let starts: Vec<i64> = db
        .prepare("SELECT mono_ns FROM writes WHERE hex(payload) = '03FF01' ORDER BY id")?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    let markers: Vec<(i64, String)> = db
        .prepare("SELECT mono_ns, label FROM markers ORDER BY id")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    let writes: Vec<(i64, String, Vec<u8>)> = db
        .prepare("SELECT mono_ns, label, payload FROM writes ORDER BY id")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<Result<_, _>>()?;

    // One fixed monotonic→Unix mapping per recording (wall clock may jump; monotonic does not).
    let offset_ns = median(notifications.iter().map(|(m, w, _, _)| w - m).collect());
    let unix = |mono: i64| (mono + offset_ns) as f64 / 1e9;

    let (mut decoder, mut health, mut events) = (Decoder::default(), Health::default(), Vec::new());
    let mut eeg: Vec<PacketRows<[i32; 6]>> = Vec::new();
    let mut eeg_flags: Vec<[u8; 9]> = Vec::new();
    let mut eeg_break = Vec::new();
    let mut pending_break = false;
    let mut accel: Vec<PacketRows<[i16; 3]>> = Vec::new();
    let mut gyro: Vec<PacketRows<[i16; 3]>> = Vec::new();
    let mut ppg: Vec<PacketRows<[u32; 3]>> = Vec::new();
    let mut vitals = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut battery = (Vec::new(), Vec::new(), Vec::new());
    let mut contact = (Vec::new(), Vec::new(), Vec::new());
    let mut status = (Vec::new(), Vec::new(), Vec::new());
    for (mono, _, uuid, bytes) in &notifications {
        let Some(c) = uuid::Uuid::parse_str(uuid)
            .ok()
            .and_then(|u| Characteristic::from_uuid(u.as_u128()))
        else {
            continue;
        };
        let last_start = starts
            .iter()
            .rev()
            .find(|s| **s <= *mono)
            .copied()
            .unwrap_or(i64::MIN / 2);
        decoder.handle(*mono, c, bytes, last_start, &mut health, &mut events);
        events.clear();
        let arrival = unix(*mono);
        match Packet::decode(c, bytes) {
            Ok(Packet::Eeg(e)) => {
                eeg_break.push(std::mem::take(&mut pending_break));
                eeg_flags.push(e.flags);
                eeg.push(PacketRows {
                    arrival,
                    mono: *mono,
                    device_time: e.device_time,
                    sequence: Some(e.sequence),
                    samples: e.samples.to_vec(),
                });
            }
            Err(frenz_proto::ParseError::EegFrameType(_) | frenz_proto::ParseError::EegUnsequenced) => {
                pending_break = true
            }
            Ok(Packet::Accelerometer(m)) => accel.push(PacketRows {
                arrival,
                mono: *mono,
                device_time: m.device_time,
                sequence: None,
                samples: m.samples.to_vec(),
            }),
            Ok(Packet::Gyroscope(m)) => gyro.push(PacketRows {
                arrival,
                mono: *mono,
                device_time: m.device_time,
                sequence: None,
                samples: m.samples.to_vec(),
            }),
            Ok(Packet::Ppg(p)) => ppg.push(PacketRows {
                arrival,
                mono: *mono,
                device_time: p.device_time,
                sequence: None,
                samples: p.samples().to_vec(),
            }),
            Ok(Packet::HeartRate(v) | Packet::Spo2(v)) => {
                vitals.0.push(arrival);
                vitals.1.push(i64::from(v.device_time));
                vitals.2.push(if c == Characteristic::HeartRate {
                    "heart_rate".to_owned()
                } else {
                    "spo2".to_owned()
                });
                vitals.3.push(v.value);
                vitals.4.push(v.check);
            }
            Ok(Packet::Battery(b)) => {
                battery.0.push(arrival);
                battery.1.push(i64::from(b.device_time));
                battery.2.push(b.percent);
            }
            Ok(Packet::Contact(k)) => {
                contact.0.push(arrival);
                contact.1.push(i64::from(k.device_time));
                contact.2.push(k.state);
            }
            Ok(Packet::Status(s)) => {
                status.0.push(arrival);
                status.1.push(format!("{:?}", s.kind()));
                status
                    .2
                    .push(s.bytes().iter().map(|b| format!("{b:02x}")).collect::<String>());
            }
            Ok(Packet::Response(r)) => {
                status.0.push(arrival);
                status.1.push(format!("response_0x{:02x}", r.opcode));
                status
                    .2
                    .push(r.payload().iter().map(|b| format!("{b:02x}")).collect::<String>());
            }
            Err(_) => {}
        }
    }

    let mut session = Session::new(
        json!({"kind": "openfrenz-recording", "file": path.file_name().map(|f| f.to_string_lossy().to_string())}),
    );
    session
        .meta
        .insert("recording_meta".into(), Value::Object(recording_meta));
    session.meta.insert("clock".into(), json!({"mono_to_unix_offset_ns": offset_ns, "time_column": "estimated acquisition time, Unix seconds (host clock)"}));
    session.meta.insert("health".into(), health_json(&health));
    let mut streams = Map::new();

    // EEG
    if !eeg.is_empty() {
        let (mut lost, mut restarts) = (0, 0);
        let ix = index_and_time(&eeg, &eeg_break, &starts, 125.0, &mut lost, &mut restarts);
        let slot = |s: usize| -> Vec<i32> {
            eeg.iter()
                .flat_map(|p| p.samples.iter().map(move |r| r.get(s).copied().unwrap_or(0)))
                .collect()
        };
        let inactive_nonzero = [2usize, 5].iter().any(|s| slot(*s).iter().any(|v| *v != 0));
        let mut table = Table::default()
            .with("t", Column::F64(ix.t))
            .with("segment", Column::I64(ix.segment))
            .with("packet", Column::I64(ix.packet))
            .with("sample", Column::I8(ix.sample))
            .with(
                "sequence",
                Column::I16(
                    eeg.iter()
                        .flat_map(|p| std::iter::repeat_n(i16::from(p.sequence.unwrap_or(0)), p.samples.len()))
                        .collect(),
                ),
            )
            .with(
                "device_time",
                Column::I64(
                    eeg.iter()
                        .flat_map(|p| std::iter::repeat_n(i64::from(p.device_time), p.samples.len()))
                        .collect(),
                ),
            )
            .with(
                "arrival",
                Column::F64(
                    eeg.iter()
                        .flat_map(|p| std::iter::repeat_n(p.arrival, p.samples.len()))
                        .collect(),
                ),
            );
        for (slot_index, name) in frenz_proto::EegPacket::ACTIVE_CHANNELS {
            table = table.with(name, Column::I32(slot(slot_index)));
        }
        if inactive_nonzero {
            table = table
                .with("slot2", Column::I32(slot(2)))
                .with("slot5", Column::I32(slot(5)));
        }
        table = table
            .with(
                "flag",
                Column::U8(eeg_flags.iter().flat_map(|f| f.iter().copied()).collect()),
            )
            .with("duplicate", Column::Bool(ix.duplicate.clone()))
            .with("held", Column::Bool(ix.held.clone()));
        let duplicates = ix.duplicate.iter().filter(|d| **d).count() / 9;
        streams.insert("eeg".into(), json!({
            "nominal_rate_hz": 125.0, "units": "raw ADC counts (µV scale unknown)", "channels": eeg_channel_json(),
            "inactive_slots_nonzero": inactive_nonzero, "segments": ix.fits.iter().map(ClockFit::to_json).collect::<Vec<_>>(),
            "frames_lost_within_segments": lost, "counter_restarts": restarts, "duplicate_packets": duplicates,
            "held_packets": ix.held.iter().filter(|h| **h).count() / 9,
        }));
        session.tables.insert("eeg", table);
    }
    // Accelerometer and gyroscope
    for (name, packets) in [("accel", &accel), ("gyro", &gyro)] {
        if packets.is_empty() {
            continue;
        }
        let (mut lost, mut restarts) = (0, 0);
        let breaks = vec![false; packets.len()];
        let ix = index_and_time(packets, &breaks, &starts, 50.0, &mut lost, &mut restarts);
        let axis = |a: usize| -> Vec<i16> {
            packets
                .iter()
                .flat_map(|p| p.samples.iter().map(move |s| s.get(a).copied().unwrap_or(0)))
                .collect()
        };
        let duplicates = ix.duplicate.iter().filter(|d| **d).count() / 25;
        let table = Table::default()
            .with("t", Column::F64(ix.t))
            .with("segment", Column::I64(ix.segment))
            .with("packet", Column::I64(ix.packet))
            .with("sample", Column::I8(ix.sample))
            .with(
                "device_time",
                Column::I64(
                    packets
                        .iter()
                        .flat_map(|p| std::iter::repeat_n(i64::from(p.device_time), 25))
                        .collect(),
                ),
            )
            .with(
                "arrival",
                Column::F64(
                    packets
                        .iter()
                        .flat_map(|p| std::iter::repeat_n(p.arrival, 25))
                        .collect(),
                ),
            )
            .with("x", Column::I16(axis(0)))
            .with("y", Column::I16(axis(1)))
            .with("z", Column::I16(axis(2)))
            .with("duplicate", Column::Bool(ix.duplicate))
            .with("held", Column::Bool(ix.held.clone()));
        let mut info = motion_json(name == "accel");
        if let Some(obj) = info.as_object_mut() {
            obj.insert(
                "segments".into(),
                json!(ix.fits.iter().map(ClockFit::to_json).collect::<Vec<_>>()),
            );
            obj.insert("duplicate_packets".into(), json!(duplicates));
            obj.insert(
                "held_packets".into(),
                json!(ix.held.iter().filter(|h| **h).count() / 25),
            );
        }
        streams.insert(name.into(), info);
        session.tables.insert(name, table);
    }
    // PPG
    if !ppg.is_empty() {
        let (mut lost, mut restarts) = (0, 0);
        let breaks = vec![false; ppg.len()];
        let ix = index_and_time(&ppg, &breaks, &starts, 25.0, &mut lost, &mut restarts);
        let ch = |c: usize| -> Vec<u32> {
            ppg.iter()
                .flat_map(|p| p.samples.iter().map(move |s| s.get(c).copied().unwrap_or(0)))
                .collect()
        };
        let table = Table::default()
            .with("t", Column::F64(ix.t))
            .with("segment", Column::I64(ix.segment))
            .with("packet", Column::I64(ix.packet))
            .with("sample", Column::I8(ix.sample))
            .with(
                "device_time",
                Column::I64(
                    ppg.iter()
                        .flat_map(|p| std::iter::repeat_n(i64::from(p.device_time), p.samples.len()))
                        .collect(),
                ),
            )
            .with(
                "arrival",
                Column::F64(
                    ppg.iter()
                        .flat_map(|p| std::iter::repeat_n(p.arrival, p.samples.len()))
                        .collect(),
                ),
            )
            .with("tag1", Column::U32(ch(0)))
            .with("tag2", Column::U32(ch(1)))
            .with("tag3", Column::U32(ch(2)))
            .with("duplicate", Column::Bool(ix.duplicate))
            .with("held", Column::Bool(ix.held.clone()));
        let held_samples = ix.held.iter().filter(|h| **h).count();
        streams.insert("ppg".into(), json!({
            "held_samples": held_samples,
            "nominal_rate_hz": 25.0, "units": "raw 19-bit optical counts",
            "channels": {"tag1": "always 0 on verified hardware", "tag2": "optical channel (LED unverified)", "tag3": "optical channel (LED unverified)"},
            "segments": ix.fits.iter().map(ClockFit::to_json).collect::<Vec<_>>(),
        }));
        session.tables.insert("ppg", table);
    }
    if !vitals.0.is_empty() {
        session.tables.insert(
            "vitals",
            Table::default()
                .with("t", Column::F64(vitals.0))
                .with("device_time", Column::I64(vitals.1))
                .with("kind", Column::Str(vitals.2))
                .with("value", Column::U8(vitals.3))
                .with("check", Column::U8(vitals.4)),
        );
    }
    if !battery.0.is_empty() {
        session.tables.insert(
            "battery",
            Table::default()
                .with("t", Column::F64(battery.0))
                .with("device_time", Column::I64(battery.1))
                .with("percent", Column::U8(battery.2)),
        );
    }
    if !contact.0.is_empty() {
        session.tables.insert(
            "contact",
            Table::default()
                .with("t", Column::F64(contact.0))
                .with("device_time", Column::I64(contact.1))
                .with("state", Column::U8(contact.2)),
        );
    }
    if !status.0.is_empty() {
        session.tables.insert(
            "status",
            Table::default()
                .with("t", Column::F64(status.0))
                .with("kind", Column::Str(status.1))
                .with("hex", Column::Str(status.2)),
        );
    }
    let mut mt = (Vec::new(), Vec::new());
    for (m, label) in &markers {
        mt.0.push(unix(*m));
        mt.1.push(label.clone());
    }
    for (m, label, bytes) in &writes {
        mt.0.push(unix(*m));
        mt.1.push(format!(
            "write:{label}:{}",
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
        ));
    }
    if !mt.0.is_empty() {
        let mut order: Vec<usize> = (0..mt.0.len()).collect();
        order.sort_by(|a, b| {
            mt.0.get(*a)
                .copied()
                .unwrap_or(0.0)
                .total_cmp(&mt.0.get(*b).copied().unwrap_or(0.0))
        });
        session.tables.insert(
            "markers",
            Table::default()
                .with(
                    "t",
                    Column::F64(order.iter().filter_map(|i| mt.0.get(*i).copied()).collect()),
                )
                .with(
                    "label",
                    Column::Str(order.iter().filter_map(|i| mt.1.get(*i).cloned()).collect()),
                ),
        );
    }
    session.meta.insert("streams".into(), Value::Object(streams));
    Ok(session)
}

/// What a recording holds, without exporting it.
#[derive(Debug, Clone)]
pub struct Summary {
    /// Time from the first to the last notification (s).
    pub seconds: f64,
    /// Samples per stream (`eeg`, `accelerometer`, `gyroscope`, `ppg`).
    pub samples: std::collections::BTreeMap<&'static str, u64>,
    /// The counters a live connection would have ended with.
    pub health: Health,
}

/// Reads a recording and counts what is in it.
pub fn summarize(path: &Path) -> Result<Summary, Error> {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let starts: Vec<i64> = db
        .prepare("SELECT mono_ns FROM writes WHERE hex(payload) = '03FF01' ORDER BY id")?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    let mut statement = db.prepare("SELECT mono_ns, char, payload FROM notifications ORDER BY id")?;
    let mut rows = statement.query([])?;
    let (mut decoder, mut health, mut out) = (Decoder::default(), Health::default(), Vec::new());
    let (mut first, mut last) = (i64::MAX, i64::MIN);
    let mut samples = std::collections::BTreeMap::<&'static str, u64>::new();
    while let Some(row) = rows.next()? {
        let t: i64 = row.get(0)?;
        let uuid: String = row.get(1)?;
        let payload: Vec<u8> = row.get(2)?;
        let Some(c) = uuid::Uuid::parse_str(&uuid)
            .ok()
            .and_then(|u| Characteristic::from_uuid(u.as_u128()))
        else {
            continue;
        };
        let last_start = starts.iter().rev().find(|s| **s <= t).copied().unwrap_or(i64::MIN / 2);
        decoder.handle(t, c, &payload, last_start, &mut health, &mut out);
        for event in out.drain(..) {
            if let frenz::Event::Packet { packet, .. } = event {
                let (name, n) = match packet.as_ref() {
                    Packet::Eeg(_) => ("eeg", 9),
                    Packet::Accelerometer(_) => ("accelerometer", 25),
                    Packet::Gyroscope(_) => ("gyroscope", 25),
                    Packet::Ppg(p) => ("ppg", p.samples().len() as u64),
                    _ => continue,
                };
                *samples.entry(name).or_default() += n;
            }
        }
        first = first.min(t);
        last = last.max(t);
    }
    Ok(Summary {
        seconds: (last - first) as f64 / 1e9,
        samples,
        health,
    })
}
