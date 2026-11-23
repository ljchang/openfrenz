# openfrenz

[![CI](https://github.com/ljchang/openfrenz/actions/workflows/ci.yml/badge.svg)](https://github.com/ljchang/openfrenz/actions/workflows/ci.yml)

Open-source tools for the FRENZ brainband: connect over Bluetooth LE, stream every sensor, record
sessions losslessly, and export time-aligned data for analysis. Written in Rust, with a small Python
package for loading the exported data.

> **Unofficial.** This project is not affiliated with or endorsed by Earable. "FRENZ" is used only
> to describe compatibility.

## What you get

| Piece | What it does |
|-------|--------------|
| `frenz` (command-line tool) | `scan`, `record`, `summary`, `export`, `synth` — see the [CLI guide](docs/CLI.md) |
| [`frenz`](crates/frenz) (Rust library) | async client: connect, typed events for every stream, commands, crash-safe recording |
| [`frenz-proto`](crates/frenz-proto) | packet decoders and command encoders; `no_std`, no dependencies, never panics on any input |
| [`frenz-export`](crates/frenz-export) | recordings → one Parquet file per stream with an estimated acquisition time for every sample |
| [`openfrenz`](python) (Python) | load exported sessions into pandas; build aligned fixed-length windows |

Streams: EEG (4 channels, 125 Hz), accelerometer and gyroscope (50 Hz), PPG (25 Hz), heart rate,
SpO2, battery and electrode-contact quality. What is known about the hardware and its Bluetooth
interface is written up in [docs/HARDWARE.md](docs/HARDWARE.md).

## Install

**Command-line tool.** Needs [Rust](https://rustup.rs) 1.88 or newer.

```sh
cargo install frenz-cli          # installs the `frenz` command
```

- macOS: the terminal application needs Bluetooth permission (System Settings → Privacy & Security →
  Bluetooth); macOS asks the first time you scan.
- Linux: install the build packages first (`sudo apt-get install libdbus-1-dev pkg-config` on
  Debian/Ubuntu), and BlueZ must be running.

**Python package.** Needs Python 3.10 or newer.

```sh
pip install openfrenz            # or: uv add openfrenz
```

## Quick start

```sh
frenz scan                       # find bands nearby
frenz record FRENZA12            # record until Ctrl-C (use the name `scan` printed)
frenz summary recordings/FRENZA12-1800000000.sqlite
frenz export  recordings/FRENZA12-1800000000.sqlite -o sessions/
```

No band at hand? `frenz synth demo.sqlite` writes a synthetic recording you can summarize and
export the same way.

```python
from openfrenz import load_session

s = load_session("sessions/FRENZA12-1800000000")
t, eeg, segment = s.signal("eeg")      # eeg: (4, n) raw counts; t: Unix seconds
```

From Rust:

```rust
use frenz::{Band, ConnectOptions, Event};
use frenz::proto::Packet;

let band = Band::connect("FRENZA12", ConnectOptions::default()).await?;
let mut events = band.subscribe();
while let Ok(event) = events.recv().await {
    if let Event::Packet { packet, .. } = event {
        if let Packet::Eeg(eeg) = packet.as_ref() {
            println!("{:?}", eeg.samples[0]);
        }
    }
}
```

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
commands, and conversion of EEG counts to microvolts (the scale
is not yet known). Recordings contain physiological data: keep them out of version control.

## Development

```sh
cargo test --workspace           # includes an end-to-end test on a synthetic recording
cargo clippy --workspace --all-targets -- -D warnings
cd python && uv venv && uv pip install -e '.[test]' && pytest
```

## License

[MIT](LICENSE)
