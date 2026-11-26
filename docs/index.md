# openfrenz

Open-source tools for the FRENZ brainband: connect over Bluetooth LE, stream every sensor, record
sessions losslessly, and export time-aligned data for analysis. One Rust client, usable from a
terminal, from Python and from Rust.

!!! note "Unofficial"
    This project is not affiliated with or endorsed by Earable. "FRENZ" is used only to describe
    compatibility.

## Pick your route

| You want to | Use | Install | Guide |
|---|---|---|---|
| Record and export from a terminal | the `frenz` command | `pip install openfrenz` | [Command line](CLI.md) |
| Read live data or analyse sessions in Python | the `openfrenz` package | `pip install openfrenz` | [Python](PYTHON.md) |
| Build the band into a Rust program | the `frenz` crate | `cargo add frenz` | [Rust](RUST.md) |
| Filter signals, compute features or run a model, offline or live | `openfrenz.dsp`, `openfrenz.Pipeline`, the `frenz-model` crate | `pip install openfrenz` | [Models and signal processing](MODELS.md) |
| Build pipelines as a graph and stream the band to LSL from a desktop app | HyperStudy Bridge | [download](https://github.com/hyperstudyio/hyperstudy-bridge/releases) | [HyperStudy Bridge](BRIDGE.md) |

All three run the same code and write the same files.

## Install

```sh
pip install openfrenz            # or: uv add openfrenz
```

This installs the Python package and the `frenz` command. It needs Python 3.10 or newer and
nothing else: wheels are provided for macOS, Linux and Windows.

## First recording

Close the band's phone app, switch the band on and put it on. Then:

```sh
frenz scan                       # prints the band's name, e.g. FRENZA12
frenz record FRENZA12            # Ctrl-C to stop
frenz export recordings/FRENZA12-*.sqlite -o sessions/
```

or from Python:

```python
import openfrenz

with openfrenz.connect("FRENZA12") as band:
    for chunk in band.stream("eeg"):
        print(chunk.data.shape, chunk.t[-1])
        break

session = openfrenz.export(band.recording)
```

No band at hand? `frenz synth demo.sqlite` writes a synthetic recording to try everything on.

## What the band sends

EEG (4 channels, 125 Hz), accelerometer and gyroscope (50 Hz), PPG (25 Hz), heart rate, SpO2,
battery and electrode-contact quality. Values are raw device counts; what is known about the
hardware is in the [hardware notes](HARDWARE.md).

## Principles

- **Nothing is silently changed.** Recordings store every packet exactly as received. Lost
  packets, repeated blocks, frozen sensor output and disconnects are reported and flagged, never
  dropped, filled in or smoothed over.
- **Honest timing.** Arrival times, the band's own clock and estimated acquisition times are kept
  separately, and the estimate's measured jitter is written into each export.
- **Raw units.** Values stay in device counts, with known scales and open questions documented.

## More

- [Source and issues on GitHub](https://github.com/ljchang/openfrenz)
- [Changelog](https://github.com/ljchang/openfrenz/blob/main/CHANGELOG.md)
- Rust API reference on docs.rs: [`frenz`](https://docs.rs/frenz),
  [`frenz-proto`](https://docs.rs/frenz-proto), [`frenz-export`](https://docs.rs/frenz-export),
  [`frenz-model`](https://docs.rs/frenz-model)
