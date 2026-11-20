# FRENZ brainband: hardware and Bluetooth interface

What this project knows about the band, from recordings and bench tests with one unit (firmware
`X8.2.2_1.1.19`, GATT firmware revision `20.3`). Each statement is tagged:

- **[V]** verified by measurement on hardware
- **[I]** inferred from indirect evidence; treat as a working assumption

Corrections and results from other units or firmware versions are welcome. All multi-byte integers
are big-endian.

## Sensors and contacts

| Part | Details |
|------|---------|
| EEG | 4 channels, 125 Hz, signed 24-bit ADC counts [V]. Microvolt scale not yet known. |
| Forehead pads | 4 dry pads. Two are recording sites: left of centre and right of centre (beside the optical sensor) [V]. The centre pad behaves like the reference and the remaining pad is presumably bias/ground [I]. |
| Side combs | 2 soft dry comb electrodes, one per side of the head, both recording sites [V]. |
| Optical sensor | PPG, 3 tagged channels at ~25 Hz, of which one is always zero [V]. Which LED each tag is: unknown. |
| Motion | 3-axis accelerometer and 3-axis gyroscope, 50 Hz each, signed 16-bit counts [V]. |
| Derived | Heart rate and SpO2 computed on the band, ~1 Hz [V]. |
| Status | Battery percentage, electrode contact quality (0–3) [V]. |
| Audio | Bone-conduction speakers and an audio chipset are present; not characterized. |

### EEG channel map [V]

Packets carry six channel slots; four carry signal.

| Slot | Name in this project | Electrode | Typical content |
|------|----------------------|-----------|-----------------|
| 0 | `forehead_right` | right forehead pad | blinks, jaw EMG |
| 1 | `comb_left` | left side comb | ~10 Hz alpha with eyes closed; large DC offset; strong mains pickup |
| 2 | — | unused, always 0 | |
| 3 | `forehead_left` | left forehead pad | blinks (strongest), jaw EMG |
| 4 | `comb_right` | right side comb | as `comb_left` |
| 5 | — | unused, always 0 | |

The map comes from pressing on the band over each contact in turn (the responding channel was 13–72×
stronger than the next) and from eyes-open/eyes-closed recordings (alpha power ×1.6–1.8 on the combs).

### Motion sensor frame and scale [V]

- Axes: **+x down** (toward the chin when worn), **+y toward the wearer's left**, **+z forward**. The
  y/z axes are rotated about 30° around x relative to the band.
- Accelerometer: ±2 g, 16-bit, ≈ 16 384 counts/g. A six-orientation static calibration of one unit gave
  per-axis scales within 1.3 % of nominal and biases of a few hundred counts; calibrate each unit.
- Gyroscope: small at-rest bias (tens to ~100 counts). Scale is probably 65.5 counts per deg/s (±500 deg/s): hand-made quarter turns gave 50–62 [I]; a controlled rotation is needed to confirm.

## Bluetooth LE interface

Band-specific services use the UUID suffix `-0000-ffff-ff45-415241424c45`. No pairing key or
payload encryption is involved [V].

| UUID prefix | Properties | Role |
|-------------|------------|------|
| `45420101` | write without response | commands |
| `45420102` | notify | command responses |
| `45420103` | notify | status messages |
| `45420201` | notify | EEG |
| `45420202` | notify | accelerometer |
| `45420203` | notify | PPG |
| `45420204` | notify | heart rate |
| `45420205` | notify | SpO2 |
| `45420206` | notify | battery |
| `45420207` | notify | gyroscope |
| `45420208` | notify | contact quality |

The band also exposes Google Fast Pair (`fe2c`), audio-chipset services and the standard battery
(`180f`) and firmware-revision (`2a26`) characteristics. **Subscribe only to the `4542…`
characteristics:** subscribing to every notifying characteristic on the device dropped the
connection [V].

### Commands and responses

A response is `[opcode, status, payload…]` with status `00` for success [V].

| Command | Bytes | Response payload |
|---------|-------|------------------|
| Firmware version | `01` | ASCII string |
| Battery | `02` | percent, u8 |
| Stream control | `03 <mask> 01` start, `03 ff 00` stop | ack. Mask bits [V]: `01` EEG, `02` accelerometer + gyroscope, `10` contact, heart rate and SpO2. `ff` starts everything. The PPG bit is not yet identified (PPG takes ~6 s to start). |
| Serial number | `06` | ASCII |
| Time sync | `07` + Unix seconds (u32) | ack; packet timestamps become Unix seconds |
| Device name | `09` | ASCII |
| Read device time | `0b` | u32 seconds |
| Device ID | `0c` | ASCII (the Bluetooth address) |
| Session notice | `0e 00` at start, `0e 03` at end | ack; effect not characterized |
| Periodic heart rate | `10 01` | ack; HR/SpO2 change from every 5 s to every 1 s |
| Prompt volume | `36` + u8 | ack |
| IMU calibration report | `f8` | 3 × u8 `std`, 3 × u16 `diff`; exact meaning [I] |

Start sequence used by this project [V]: firmware version, time sync, prompt volume 0, periodic heart
rate, session notice, stream start, IMU calibration report. Stop: stream stop, session-end notice.

The device has further opcodes (audio, alarms, stimulation, file transfer, firmware update, test
modes, reset, rename). They are intentionally not implemented here until characterized on a bench.

### Stream payloads

Timestamps are whole seconds: seconds since power-on until a time sync, Unix seconds afterwards [V].

| Stream | Layout |
|--------|--------|
| EEG | 178 bytes: sequence u8, type u8 (= 6), seconds u32, then 9 samples × (6 × i24 + flag u8), trailer u8. 13.9 packets/s (125.0 samples/s measured). Flag is 1 on ~4 % of samples and the trailer varies 0–16; meanings unknown. |
| Accelerometer, gyroscope | 158 bytes: seconds u32, 25 × (x, y, z as i16), 4 trailing bytes (mostly zero; meaning unknown). 2 packets/s. |
| PPG | seconds u32, then 12 or 13 samples × three 24-bit words, then 2 zero bytes (114 or 123 bytes). Each word: tag in bits 23–19 (1, 2, 3), unsigned 19-bit value in bits 18–0. 2 packets/s. |
| Heart rate, SpO2 | 6 bytes: seconds u32, one byte of unknown meaning, value u8. |
| Battery | 5 bytes: seconds u32, percent u8. Every 5 s. |
| Contact | 5 bytes: seconds u32, state u8 (0–3). Every 2 s. |
| Status | variable. `0e 00 00 00 00` about once a second. Every 60 s a block: `07 12 ff ee` + u16 that tracks battery voltage in mV (4039 → 4010 as charge fell 90 → 87 %) [I], `07 12 aa aa` + u16 (constant 37), `07 12 ff ff` + u16 (44–62, unknown), `07 11`. `07 db`, `07 d0` are sent when the band is about to power off, whether by timeout or by its power button [V]. |

## Behaviour worth knowing

- **Streaming state persists across connections** [V]. After a session that sent the stop command, no
  EEG/motion/PPG/contact data arrives until stream start is sent again (heart rate, SpO2, battery and
  status still do). If the previous session did not stop the streams, data flows as soon as the
  characteristics are subscribed. Always send stream start.
- **The EEG sequence counter restarts at 0** when stream start arrives while streams are running [V].
- **Start-up artifacts** [V]: about 0.8 s after every stream start, one EEG packet repeats the
  previous packet's samples exactly while the sequence counter still advances. Once (1 of 5
  sessions) a block was repeated three times, the last copies arriving as frames without the leading
  sequence byte.
- **Contact quality settles slowly** [V]: after a stream start it reads 0, 0, 1, 1, 3 at two-second
  intervals. 3 is good contact; it flickers to 2 with marginal contact and drops to 0 when the band is
  taken off.
- **PPG can freeze** [V]: with marginal contact the PPG output held one constant value for 2–5 s and
  the band sent extra PPG packets per second during the freeze. PPG also switches off ~4 s after the
  band is removed.
- **Automatic power-off** [V, one observation each]: while streaming, the band powered off ~33 s after
  being taken off (putting it back on within ~26 s kept it on); while not streaming it powered off
  ~11 minutes after the last disconnect.
- **Delivery jitter** [V]: Bluetooth delivery delay beyond the minimum had a 95th percentile of
  80–100 ms for EEG and motion and ~250–300 ms for PPG on macOS. Estimate sample times from sample
  counts, not arrival times (see `frenz-export`).
- **Streams are aligned** [V]: taps on the band appear in the accelerometer and in the EEG within
  ±20 ms (one motion sample) of each other in exported time. EEG packets are stamped ~40 ms later than
  motion packets on the band's own whole-second clock, which is a labelling difference, not a delay.
- **Measured sample rates** [V]: EEG 125.00 Hz, motion 50.00 Hz, PPG 25.00 Hz, each within 0.01 Hz over
  three minutes.

## Open questions

- EEG microvolt scale.
- The four trailing bytes of motion packets; gyroscope scale.
- Which optical channel is which LED, and why one is zero.
- The PPG stream-control bit, the remaining status values and the IMU calibration report.
- The EEG per-sample flag clusters in the last packet of each device second and the trailer is an
  8-bit mask; both look timing-related, exact meaning unknown.
- Behaviour on other firmware versions, sessions longer than an hour, and several bands at once.
