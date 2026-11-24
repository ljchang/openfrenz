# Changelog

## Unreleased

- **`frenz-model`: run models on recordings and live bands.** A `Model` names the streams it
  reads (any the band sends; required, preferred, or taken as they arrive) and is run one step at
  a time by `run_recording` or `Live`, with a reset wherever data is missing. `Registry` loads
  model files (`config.json`, `model.safetensors`) from a folder or, with the `hub` feature, from
  the Hugging Face Hub. See the [Rust guide](docs/RUST.md#run-a-model).
- **Signal-processing blocks** (`frenz_model::dsp`): Butterworth, notch and DC-removal filters
  that carry their state across steps, zero-phase filtering, power spectra and band power,
  envelope and phase, coherence between channels, phase-amplitude coupling, summaries and
  quality checks. Tested against SciPy.
- **Pipelines: models as data.** `Pipeline` runs a model described by `config.json` alone, a list
  of blocks and how they connect, with stored numbers and small networks (dense, LSTM) read from
  `model.safetensors`. See the [Rust guide](docs/RUST.md#models-as-data-pipelines).
- Pipelines can use a stream when it is there (`"required": false`) and take slow streams as
  they arrive (`"arrived": true`); skipped steps are counted by the stream that was missing.
- More blocks: Morlet wavelets, heartbeats and heart-rate variability from the pulse sensor, and
  artifact subspace reconstruction (calibration and cleaning), each also usable in a pipeline.
- **Python: `openfrenz.dsp` and `openfrenz.Pipeline`.** The same blocks on NumPy arrays, and
  pipelines loaded from a folder or the Hugging Face Hub that score a recording (`run`) or a band
  as it streams (`stream`). See [Models and signal processing](docs/MODELS.md).
- The minimum Rust version is 1.89, which a dependency already required.
- The live sample buffers moved from the Python bindings into `frenz_model::streams`, so Rust
  programs get the same timed, segmented samples Python does.

## 0.3.0

- **Python bindings.** `pip install openfrenz` now gives the whole client, not only the loader:
  `scan`, `connect`, `replay`; live streams as NumPy chunks with `to_pandas()` and `to_polars()`;
  events; commands by name; `export` and `summary`. Recording is on by default. The wheel also
  installs the `frenz` command. See the [Python guide](docs/PYTHON.md).
- **Fix: recordings stopped after 10 minutes.** The band powers itself off 10 minutes after the
  stream start unless it receives a session-running notice (`0e 02`). The start sequence now sends
  it (`Command::StartSession`).
- **Fix: heart rate and SpO2 values.** The reading is byte 4 of the packet and byte 5 flags a
  missing reading; the two were swapped, so `value` held the flag and `check` the reading. Exports
  of older recordings come out corrected when re-exported.
- Replay (`Band::replay`): play a recording back through the same path live data takes. It
  reproduces recorded stream starts.
- `ConnectOptions::notify` selects which characteristics to subscribe to.
- `frenz_export::recording::summarize`; segment rules shared between exports and live use
  (`Segmenter`).
- `frenz-cli` is now also a library (`frenz_cli::run`).

## 0.2.1

- Reconnection accepts only the same physical device as the original connection, not any device
  advertising the same name.

## 0.2.0

- **Automatic reconnection.** If the link is lost, `Band` reconnects, re-runs the start sequence and
  keeps subscribers and recorders attached. New events `Reconnecting`, `Reconnected`,
  `ReconnectFailed`; `ConnectOptions::reconnect`; `frenz record --no-reconnect` and
  `--reconnect-seconds`. `Band::info()` now returns an owned value.
- Export: a pause in a stream always starts a new segment (previously a stream restarted after a
  pause could be fitted as one segment with a wrong rate).
- `frenz::synth::write_recording_with_pause` for testing.
- Hardware notes: stream-control mask bits, battery-voltage status value, stream alignment measured
  with a tap test, tentative gyroscope scale.

## 0.1.0

Initial release: protocol decoders, Bluetooth LE client with crash-safe recording, Parquet export
with estimated sample times, the `frenz` command-line tool and the Python loader.
