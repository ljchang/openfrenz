# openfrenz

[![CI](https://github.com/ljchang/openfrenz/actions/workflows/ci.yml/badge.svg)](https://github.com/ljchang/openfrenz/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openfrenz)](https://pypi.org/project/openfrenz/)
[![crates.io](https://img.shields.io/crates/v/frenz)](https://crates.io/crates/frenz)
[![docs.rs](https://img.shields.io/docsrs/frenz)](https://docs.rs/frenz)

**Documentation: <https://ljchang.github.io/openfrenz/>**

Open-source tools for the FRENZ brainband: connect over Bluetooth LE, stream every sensor, record
sessions losslessly, and export time-aligned data for analysis. One Rust client, usable from a
terminal, from Python and from Rust.

> **Unofficial.** This project is not affiliated with or endorsed by Earable. "FRENZ" is used only
> to describe compatibility.

Streams: EEG (4 channels, 125 Hz), accelerometer and gyroscope (50 Hz), PPG (25 Hz), heart rate,
SpO2, battery and electrode-contact quality.

## Pick your route

| You want to | Use | Install | Guide |
|---|---|---|---|
| Record and export from a terminal | the `frenz` command | `pip install openfrenz` | [CLI guide](docs/CLI.md) |
| Read live data or analyse sessions in Python | the `openfrenz` package | `pip install openfrenz` | [Python guide](docs/PYTHON.md) |
| Build the band into a Rust program | the `frenz` crate | `cargo add frenz` | [Rust guide](docs/RUST.md) |

All three run the same code and write the same files.

## Install

```sh
pip install openfrenz            # or: uv add openfrenz
```

This installs the Python package and the `frenz` command. It needs Python 3.10 or newer and
nothing else: wheels are provided for macOS (Apple silicon and Intel), Linux (x86-64 and ARM64) and
Windows.

If you prefer to install the command with Rust (1.88 or newer): `cargo install frenz-cli`. On
Linux that needs the D-Bus development files first (`sudo apt-get install libdbus-1-dev pkg-config`
on Debian/Ubuntu).

- macOS: the application you run from (Terminal, VS Code, …) needs Bluetooth permission (System
  Settings → Privacy & Security → Bluetooth); macOS asks the first time you scan.
- Linux: BlueZ must be running.

## Before you connect

- Close the band's phone app, and anything else connected to the band: it accepts one connection.
- Switch the band on and wear it. Electrode contact takes about 10 seconds to settle.
- The band powers itself off about half a minute after it is taken off while streaming. It then
  needs its button pressed to come back.

## Quick start

From a terminal:

```sh
frenz scan                       # find bands nearby
frenz record FRENZA12            # record until Ctrl-C (use the name `scan` printed)
frenz summary recordings/FRENZA12-1800000000.sqlite
frenz export  recordings/FRENZA12-1800000000.sqlite -o sessions/
```

From Python:

```python
import openfrenz

with openfrenz.connect("FRENZA12") as band:       # records to recordings/ by default
    for chunk in band.stream("eeg"):
        chunk.data                                # (4, n) raw counts, NumPy
        chunk.t                                   # (n,) estimated Unix seconds
        chunk.to_pandas()                         # or chunk.to_polars()
        break

session = openfrenz.export(band.recording)        # same files as `frenz export`
t, eeg, segment = session.signal("eeg")
```

From Rust (`cargo add frenz` and `cargo add tokio --features macros,rt-multi-thread`):

```rust
use frenz::proto::Packet;
use frenz::{Band, ConnectOptions, Event};

#[tokio::main]
async fn main() -> Result<(), frenz::Error> {
    let band = Band::connect("FRENZA12", ConnectOptions::default()).await?;
    let mut events = band.subscribe();
    while let Ok(event) = events.recv().await {
        if let Event::Packet { packet, .. } = event {
            if let Packet::Eeg(eeg) = packet.as_ref() {
                println!("{:?}", eeg.samples[0]);
            }
        }
    }
    Ok(())
}
```

No band at hand? `frenz synth demo.sqlite` writes a synthetic recording that you can summarize,
export, and replay from Python (`openfrenz.replay`) or Rust (`Band::replay`) as if it were live.

## What you get on disk

- **A recording** (`recordings/NAME-TIME.sqlite`): one file with every packet the band sent and
  every command it was sent, exactly as transmitted, with arrival times and your markers. It is
  written continuously and survives a crash.
- **A session** (`sessions/NAME-TIME/`, made by `export`): one Parquet file per stream and a
  `session.json`. Every sample has an estimated acquisition time in Unix seconds, so streams line
  up directly. Any Parquet reader opens them (pandas, polars, Arrow, DuckDB).

Values are raw device counts. EEG is not converted to microvolts because the scale is not known;
what is known about the hardware is in [docs/HARDWARE.md](docs/HARDWARE.md). The session format is
described in the [CLI guide](docs/CLI.md).

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| `0 band(s) found` | The band is off (press its button), or still connected to the phone app or another program |
| Nothing found on macOS, band is on | The terminal or editor has no Bluetooth permission (System Settings → Privacy & Security → Bluetooth) |
| "band … not found" when connecting | The name must match what `frenz scan` prints, including capitals |
| The recording stops after about 30 seconds | The band was not on a head: it powers off when it has no electrode contact |
| EEG looks flat or noisy at the start | Contact is still settling; check that the `contact` stream reads 3 |
| The recording stops after 10 minutes | A version before 0.3.0; upgrade |

## Design principles

- **Nothing is silently changed.** Recordings store every notification exactly as received. Lost
  packets, repeated blocks, frozen sensor output and disconnects are reported and flagged, never
  dropped, filled in or smoothed over.
- **Honest timing.** Arrival times, the band's own clock and estimated acquisition times are kept
  separately; the estimate and its measured jitter are documented in each export.
- **Raw units.** Values stay in device counts, with known scales and open questions documented.

## Status and limits

Tested with one band (firmware `X8.2.2_1.1.19`) on macOS. Linux and Windows build and pass the
hardware-free tests in CI but have not been tried with a band. The live Bluetooth path (including
automatic reconnection) is tested by hand, not in CI. Not yet implemented: audio and stimulation
commands. Recordings contain physiological data: keep them out of version control.

## Repository layout

| Path | What it is |
|---|---|
| [`crates/frenz`](crates/frenz) | the client library: connect, typed events for every stream, commands, crash-safe recording, replay |
| [`crates/frenz-proto`](crates/frenz-proto) | packet decoders and command encoders; `no_std`, no dependencies, never panics on any input |
| [`crates/frenz-export`](crates/frenz-export) | recordings → Parquet sessions with an estimated time for every sample |
| [`crates/frenz-cli`](crates/frenz-cli) | the `frenz` command |
| [`crates/frenz-py`](crates/frenz-py), [`python`](python) | the Python package |

## Development

```sh
cargo test --workspace           # includes an end-to-end test on a synthetic recording
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p frenz --example replay                # no band needed
cd python && uv venv && uv pip install -e '.[test]' && .venv/bin/python -m pytest   # builds the Rust extension
uvx zensical serve                                 # preview the documentation site (built from docs/)
```

Changes are listed in the [changelog](CHANGELOG.md).

## License

[MIT](LICENSE)
