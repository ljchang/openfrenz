# Changelog

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
