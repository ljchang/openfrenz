# `frenz` Rust guide

Use the FRENZ brainband from your own Rust program: connect, receive every stream as typed
packets, send commands, record losslessly, and export. The [`frenz` command](CLI.md) and the
[Python package](PYTHON.md) are built on these crates.

| Crate | Use it for |
|---|---|
| [`frenz`](https://docs.rs/frenz) | The client: scan, connect, events, commands, recording, replay |
| [`frenz-proto`](https://docs.rs/frenz-proto) | Packet and command types (re-exported as `frenz::proto`); `no_std`, no dependencies |
| [`frenz-export`](https://docs.rs/frenz-export) | Recordings → Parquet session folders with a time for every sample |

The API reference for each is on docs.rs. Complete programs for everything below are in
[`crates/frenz/examples`](../crates/frenz/examples) and
[`crates/frenz-export/examples`](../crates/frenz-export/examples).

## Set up

```sh
cargo add frenz
cargo add tokio --features macros,rt-multi-thread,signal
```

Needs Rust 1.88 or newer. The client is async and runs on [Tokio](https://tokio.rs).

- macOS: the application you run from (Terminal, your IDE) needs Bluetooth permission; macOS asks
  the first time.
- Linux: install the D-Bus development files to build (`sudo apt-get install libdbus-1-dev
  pkg-config` on Debian/Ubuntu); BlueZ must be running.
- Windows: no extra setup.

Before connecting, close the band's phone app (the band accepts one connection), switch the band
on and wear it. It powers itself off about half a minute after it is taken off while streaming.

## Connect and read

```rust
use frenz::proto::Packet;
use frenz::{Band, ConnectOptions, Event};

#[tokio::main]
async fn main() -> Result<(), frenz::Error> {
    let band = Band::connect("FRENZA12", ConnectOptions::default()).await?;
    println!("{:?}, battery {} %", band.info().firmware, band.battery().await?);

    let mut events = band.subscribe();
    while let Ok(event) = events.recv().await {
        if let Event::Packet { packet, .. } = event {
            if let Packet::Eeg(eeg) = packet.as_ref() {
                // Nine samples per frame; slots 0, 1, 3 and 4 hold the four electrodes.
                let [forehead_right, comb_left, _, forehead_left, comb_right, _] = eeg.samples[0];
                println!("{forehead_right} {comb_left} {forehead_left} {comb_right}");
            }
        }
    }
    band.stop().await?;
    Ok(())
}
```

`frenz::scan(duration)` lists nearby bands if you do not know the name. `Band::connect` finds the
band, subscribes to its streams and sends the start sequence, so data flows as soon as it
returns. `band.stop()` stops the streams and disconnects; dropping a `Band` disconnects without
the stop commands.

### Packets

`Event::Packet` carries one of these (`frenz::proto::Packet`), with the computer's arrival time:

| Variant | Contents | Rate |
|---|---|---|
| `Eeg` | 9 samples × 6 slots of signed 24-bit counts (slots 2 and 5 unused), frame counter | 125 Hz |
| `Accelerometer`, `Gyroscope` | 25 samples × 3 axes, `i16` | 50 Hz |
| `Ppg` | up to 16 samples of tagged 19-bit values | 25 Hz |
| `HeartRate`, `Spo2` | one reading and a "no reading" flag | 1 Hz |
| `Battery`, `Contact` | percentage; contact quality 0–3 (3 is good) | every few seconds |
| `Status`, `Response` | the band's status messages and command replies | |

Values are raw device counts. What is known about scales and the electrode map is in
[HARDWARE.md](HARDWARE.md). Every packet also has the band's own clock in whole seconds
(`device_time`). Packets do not carry a time per sample; the exporter estimates one (see
[Export](#export)).

### Everything else is an event too

Nothing is fixed up silently. Besides packets, a subscriber receives:

| Event | Meaning |
|---|---|
| `SequenceGap` | EEG frames were lost over the radio |
| `CounterRestart` | The frame counter restarted after a stream start (not data loss) |
| `OffHead`, `OnHead` | Electrode contact lost or restored |
| `PowerOffWarning` | The band is about to power itself off |
| `Disconnected`, `Reconnecting`, `Reconnected`, `ReconnectFailed` | Link state |
| `DecodeError` | A notification could not be decoded (it is still recorded) |

`band.health()` returns running counts of all of these at any time.

`subscribe()` returns a Tokio broadcast receiver, so several tasks can each have their own. A
receiver that falls more than about 16,000 events behind gets `RecvError::Lagged(n)` and carries
on from the oldest event still held; handle that case if your processing can be slow. See
[`examples/stream.rs`](../crates/frenz/examples/stream.rs).

## Reconnection

If the link drops, the client reconnects by itself, sends the start sequence again and keeps every
subscriber and recorder attached. It gives up after `ConnectOptions::reconnect` (5 minutes by
default; `None` switches reconnection off) and sends `ReconnectFailed`. Only the same physical
device is accepted on reconnect.

```rust
use std::time::Duration;
use frenz::ConnectOptions;

let options = ConnectOptions {
    reconnect: Some(Duration::from_secs(60)),
    ..ConnectOptions::default()
};
```

## Commands

```rust
use frenz::proto::Command;

let reply = band.request(Command::GetSerialNumber).await?;   // waits for the reply
band.send(Command::SetPromptVolume(0)).await?;               // does not wait
let percent = band.battery().await?;
```

`request` fails with `Error::Timeout` if the band does not answer within
`ConnectOptions::command_timeout` and with `Error::CommandFailed` if it answers with an error
status. The start sequence already sends everything needed for streaming, including the notice
that keeps the band from powering off after 10 minutes; set `ConnectOptions::start` to
`StartSequence::None` to send nothing and drive the band yourself.

## Record

```rust
use frenz::{Band, ConnectOptions, Recorder};

let band = Band::connect("FRENZA12", ConnectOptions::default()).await?;
let recorder = Recorder::create("session.sqlite", &band)?;
band.mark("task_start");                    // a labelled time point in the recording

tokio::signal::ctrl_c().await?;

let health = band.stop().await?;
let summary = recorder.finish().await?;     // integrity check; leaves one self-contained file
```

A recording is one SQLite file holding every notification and every command exactly as sent and
received, with arrival times, plus your markers. It is written by its own thread and synced to
disk about ten times a second, so a crash loses at most the last moments and never corrupts the
file. `Recorder::create` refuses to overwrite an existing file. Create the recorder right after
connecting so the file starts at the connection. See
[`examples/record.rs`](../crates/frenz/examples/record.rs).

Recordings contain physiological data: keep them out of version control.

## Export

```sh
cargo add frenz-export
```

```rust
use std::path::Path;

let summary = frenz_export::recording::summarize(Path::new("session.sqlite"))?;
println!("{:.1} s, {} EEG frames lost", summary.seconds, summary.health.frames_lost);

let session = frenz_export::read(Path::new("session.sqlite"))?;
session.write(Path::new("sessions/session"))?;
```

The session folder has one Parquet file per stream (`eeg`, `accel`, `gyro`, `ppg`, `vitals`,
`battery`, `contact`, `status`, `markers`) and a `session.json` describing units, the channel map,
the timing fit and data-quality counts. Every row has `t`, an estimated acquisition time in Unix
seconds on the computer's clock, so streams line up directly. Repeated packets are flagged
`duplicate`, frozen sensor output `held`, and each unbroken stretch of data has its own `segment`
number; nothing is dropped or interpolated. The [CLI guide](CLI.md) describes
the columns, and any Parquet reader (pandas, polars, Arrow, DuckDB) opens the files.

## Develop without a band

`frenz::synth::write_recording(path, seconds)` writes a synthetic recording in the real format,
and `Band::replay(path, ReplayOptions::default())` plays any recording back through the same
decoder and event path a live band uses. A replay has no device to command, so `send` and
`request` return an error. See [`examples/replay.rs`](../crates/frenz/examples/replay.rs):

```sh
cargo run -p frenz --example replay
```

This is how the test suite runs in CI without hardware, and it is a good way to test your own
processing on a real session you recorded earlier.

## Only the decoders

`frenz-proto` has no dependencies and builds without the standard library
(`default-features = false`). If you have your own Bluetooth stack, pass each notification's
characteristic and bytes to `Packet::decode` and build commands with `Command::encode`. Decoding
never panics on any input.

## Limits

- Tested with one band (firmware `X8.2.2_1.1.19`) on macOS. Linux and Windows build and pass the
  tests that need no band but have not been tried with one.
- One connection per band.
- Audio and stimulation commands are not implemented, and EEG counts are not converted to
  microvolts (the scale is not known).
