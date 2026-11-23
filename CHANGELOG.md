# Changelog

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
