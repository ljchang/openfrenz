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
[`crates/frenz/examples`](https://github.com/ljchang/openfrenz/tree/main/crates/frenz/examples) and
[`crates/frenz-export/examples`](https://github.com/ljchang/openfrenz/tree/main/crates/frenz-export/examples).

## Set up

```sh
cargo add frenz
cargo add tokio --features macros,rt-multi-thread,signal
```

Needs Rust 1.89 or newer. The client is async and runs on [Tokio](https://tokio.rs).

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
[`examples/stream.rs`](https://github.com/ljchang/openfrenz/blob/main/crates/frenz/examples/stream.rs).

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
[`examples/record.rs`](https://github.com/ljchang/openfrenz/blob/main/crates/frenz/examples/record.rs).

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
`request` return an error. See [`examples/replay.rs`](https://github.com/ljchang/openfrenz/blob/main/crates/frenz/examples/replay.rs):

```sh
cargo run -p frenz --example replay
```

This is how the test suite runs in CI without hardware, and it is a good way to test your own
processing on a real session you recorded earlier.

## Run a model

`frenz-model` runs a model step by step, the same way on a recording and on a band. A model says
which streams it reads and how much of each makes one step, and returns named values per step:

```rust
use frenz_model::{Input, Model, ModelError, Spec, Step};

struct Mine { spec: Spec /* and whatever the model carries between steps */ }

impl Model for Mine {
    fn spec(&self) -> &Spec { &self.spec }
    fn reset(&mut self) { /* forget everything carried from earlier steps */ }
    fn step(&mut self, step: &Step) -> Result<Option<Vec<f64>>, ModelError> {
        let eeg = step.get("eeg").ok_or("no eeg")?;   // eeg.values[channel][sample], eeg.t[sample]
        Ok(Some(vec![/* one value per output name */]))
    }
}
```

The inputs in the spec can be any stream the band sends:

| Input | What a step gets |
|---|---|
| `Input::samples("eeg", 250)` | exactly 250 consecutive samples; the step is skipped if they are not all there |
| `Input::samples("ppg", 50).preferred()` | the same when the stream is there, and the step runs without it otherwise (`step.get("ppg")` is `None`) |
| `Input::arrived("heart_rate")` | whatever arrived during the step, possibly nothing; for streams without a steady rate |

The first input sets the steps and every other stream is cut to match. Stream names are those of
`frenz_model::streams::names()`: `eeg`, `accel`, `gyro`, `ppg`, `heart_rate`, `spo2`, `battery`
and `contact`. Values are raw counts.

```rust
// A recording, as fast as the model runs:
let run = frenz_model::run_recording(model, Path::new("session.sqlite"))?;   // run.outputs, run.missed

// A connected or replayed band, one output as each step completes:
let mut live = frenz_model::Live::new(model, &band, true)?;
while let Some(output) = live.next().await? { /* output.t, output.values */ }
```

The model is reset at the start, at every new segment and after every step that had to be skipped
because data was missing, so it never carries state across a hole. Skipped steps are counted by
the stream that was missing (`Live::missed`, `Run::missed`), which is how a model that stays silent
says what it is waiting for. Repeated and frozen packets
are left out. Live, pauses and lost data are recognised by when packets arrive, so replay a
recording at its recorded pace to see what a live run would have given, and use `run_recording`
to score it quickly. See [`examples/rms.rs`](https://github.com/ljchang/openfrenz/blob/main/crates/frenz-model/examples/rms.rs):

```sh
cargo run -p frenz-model --example rms
```

### Model files

A trained model is a folder holding `config.json` and usually `model.safetensors`. `config.json`
names the kind of model under `openfrenz_model`, and a program lists the kinds it was built with:

```rust
let registry = frenz_model::Registry::new()
    .with("mine", |dir| Ok(Box::new(Mine::from_dir(dir)?)));
let model = registry.load("owner/name")?;       // a folder, or a Hugging Face repository
```

Loading from the Hugging Face Hub needs the `hub` feature. A repository is written `owner/name`
or `owner/name@revision`; files are kept per commit under `~/.cache/openfrenz`, and a branch
fetched before still loads when Hugging Face cannot be reached. Private repositories use the token
in `HF_TOKEN` or the one saved by the Hugging Face command-line login; programs that keep their own
credentials pass one with `frenz_model::resolve_with(source, Some(token))` or `fetch_with`. A
repository that cannot be read gives `Error::Access`, with a message saying what to do.
The files hold numbers and settings only: the code for a kind of model has to be part of the
program that loads it.

### Blocks and pipelines

`frenz_model::dsp` has the signal-processing blocks a model is usually made of, as plain
functions and small stateful types: `Design` and `Filter` (Butterworth, notch, DC removal, with
state carried across steps), `filtfilt`, `periodogram`, `welch`, `morlet`, `analytic`,
`cross_spectrum` (coherence), `coupling`, `heart`, `asr_calibrate` and `Asr`, `summary` and
`quality`. Their numbers follow SciPy's and are tested against it.

A model made only of those blocks needs no code at all: `Pipeline::from_dir` reads it from a
`config.json`, and a registry built with `.with_pipelines()` loads it like any other kind. `Pipeline::from_config` builds one from a
config held in memory, checked the same way, for programs that write configs (such as an editor); blocks that
need stored numbers are refused there, since there is no `model.safetensors`. The
[Models and signal processing](MODELS.md) guide describes every block and the file format.

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
