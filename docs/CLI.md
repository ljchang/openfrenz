# `frenz` command-line guide

Install with `pip install openfrenz` (no Rust needed) or `cargo install frenz-cli` (or
`cargo build --release -p frenz-cli` in a checkout; the binary is `target/release/frenz`). Run `frenz --help` or `frenz <command> --help` for the options.

On macOS the terminal application needs Bluetooth permission (System Settings → Privacy & Security →
Bluetooth). On Linux, BlueZ must be running.

## Before you connect

- Close the band's phone app, or anything else connected to it: the band accepts one connection.
- Switch the band on and wear it. Electrode contact takes about 10 seconds to settle after streaming
  starts.
- The band powers itself off about half a minute after it is taken off while streaming, and after
  roughly ten minutes when idle.

## `frenz scan`

Lists nearby bands, strongest signal first.

```sh
$ frenz scan
  -56 dBm  FRENZA12
1 band(s) found
```

`--seconds N` changes the scan time (default 8).

## `frenz record <NAME>`

Connects to the band with that name, starts all streams and writes everything it sends to one
SQLite file. Stops on Ctrl-C or after `--seconds N`. If the connection is lost it reconnects
automatically and continues in the same file.

```sh
$ frenz record FRENZA12 --seconds 60
connecting to FRENZA12 …
connected: firmware Some("X8.2.2_1.1.19"), IMU calibration Some(([41, 43, 40], [3482, 3828, 2853]))
recording to recordings/FRENZA12-1800000000.sqlite (Ctrl-C to stop)
  accelerometer=12 battery=2 … eeg=88 … | gaps 0 (frames lost 0), restarts 0, irregular EEG 0, …
wrote recordings/FRENZA12-1800000000.sqlite: 1502 notifications, 9 writes, 5 markers; recorder lagged 0; integrity ok
```

| Option | Meaning |
|--------|---------|
| `-o, --output FILE` | where to write (default `recordings/<name>-<unix time>.sqlite`); an existing file is never overwritten |
| `--seconds N` | stop after N seconds |
| `--no-reconnect` | stop when the connection is lost |
| `--reconnect-seconds N` | give up reconnecting after N seconds (default 300) |
| `--start standard\|none` | commands sent after connecting; `none` sends nothing (the band then only streams if a previous session left it streaming) |

The status line printed every 5 seconds shows notification counts per stream and the health
counters:

- **gaps / frames lost**: EEG packets that never arrived (from the sequence counter).
- **restarts**: the band restarted its EEG counter after the start command; not data loss.
- **irregular EEG**: malformed EEG frames the band occasionally sends at start-up; they are stored but
  not decoded.
- **contact**: electrode contact quality, 0 (none) to 3 (good).

Messages such as "band reports no contact", "power-off warning", "connection lost" and
"reconnected" appear as they happen. A reconnection is stored as `disconnected` / `reconnected`
markers, and `export` puts the data after it in a new segment. The band switches itself off about
half a minute after being taken off, so a reconnect succeeds only if it is still on or is switched on
again in time.

The file is safe against crashes and power loss: data is committed at least every 100 ms, and on a
normal stop the file is checked and finalized (`integrity ok`).

## `frenz summary <FILE>`

Decodes a recording and reports its length, sample counts and rates per stream, and the same health
counters as above. Use it to check a session before relying on it.

```sh
$ frenz summary recordings/FRENZA12-1800000000.sqlite
recordings/FRENZA12-1800000000.sqlite: 60.1 s
  accelerometer:    3000 samples, 49.92 Hz
            eeg:    7497 samples, 124.75 Hz
  …
```

## `frenz export <FILE>… [-o DIR]`

Converts recordings into session folders for analysis (default output `sessions/`), one folder per
recording:

```
sessions/FRENZA12-1800000000/
├── session.json     units, channel map, timing fits, quality counts
├── eeg.parquet      t, segment, packet, sample, sequence, device_time, arrival,
│                    forehead_right, comb_left, forehead_left, comb_right, flag, duplicate, held
├── accel.parquet    t, segment, …, x, y, z, duplicate, held
├── gyro.parquet
├── ppg.parquet      t, segment, …, tag1, tag2, tag3, duplicate, held
├── vitals.parquet   t, kind (heart_rate | spo2), value
├── battery.parquet  contact.parquet  status.parquet
└── markers.parquet  t, label (session markers and every command sent)
```

Things to know about the tables:

- **`t`** is the estimated time each sample was acquired, in Unix seconds on the recording computer's
  clock, the same clock for every stream. It is computed from sample counts, with the stream's
  actual rate fitted per segment, so it is free of Bluetooth delivery jitter. `arrival` is when the
  packet reached the computer; `device_time` is the band's own whole-second clock.
- **`segment`** numbers each uninterrupted stretch. Time is only continuous within a segment.
- **`duplicate`** marks samples that repeat the previous packet exactly; **`held`** marks samples from
  a frozen sensor output. They are kept so you can see them; exclude them for analysis.
- Values are raw device counts. See [HARDWARE.md](HARDWARE.md) for scales and the channel map.

The command prints each stream's fitted sample rate and delivery jitter.

## `frenz synth <FILE>`

Writes a synthetic recording (`--seconds N`, default 60) in exactly the format `record` produces.
Use it to try `summary`, `export` and the Python package without a band.

## Loading exports in Python

See the [Python guide](PYTHON.md), which also covers connecting and streaming from Python.

```python
from openfrenz import load_session, find_sessions, WindowDataset

s = load_session("sessions/FRENZA12-1800000000")
s["eeg"]                               # pandas DataFrame
t, x, segment = s.signal("eeg")        # arrays with duplicate/held samples removed

windows = WindowDataset(find_sessions("sessions"), window_s=10, hop_s=5,
                        streams=("eeg", "accel", "gyro", "ppg"), min_contact=3)
windows[0]["eeg"].shape                # (4, 1250)
```

`WindowDataset` only returns windows that lie inside one segment with no gaps, and can require good
electrode contact.
