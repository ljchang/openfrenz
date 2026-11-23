# Python live API: design (draft)

Status: in progress. Steps 1 to 3 of [Steps](#steps) are implemented: everything in the API below.
Wheels for all platforms, the release and the user guide remain. Nothing here has been run
against a real band from Python yet; all tests use replayed recordings. This document describes adding live access to the band from
Python: scan, connect, stream, send commands and record, with the Rust client doing the work
underneath.

## Goals

- `pip install openfrenz` (or `uv add openfrenz`) is the whole installation. No Rust toolchain.
- A few lines in a script or notebook get live data as NumPy arrays.
- Everything the Rust client can do is reachable: commands, events, health counters, recording.
- The project's rules still hold: nothing is silently dropped, filled in or smoothed, and timing
  is reported honestly.

## Not in scope

- Lab Streaming Layer output. A user who wants it can publish chunks with `pylsl` in a few lines.
- A background service that keeps the connection alive across Python restarts (see
  [Later](#later)).
- Audio and stimulation commands (not implemented in the Rust client either).

## Sketch

```python
import openfrenz

for found in openfrenz.scan(timeout=8):
    print(found.name, found.rssi)

with openfrenz.connect("FRENZA12") as band:   # records to recordings/ by default
    print(band.info.firmware, band.battery())

    band.mark("task_start")
    for chunk in band.stream("eeg"):          # blocks until new samples arrive
        chunk.data                            # (4, n) int32, raw counts
        chunk.t                               # (n,) estimated Unix seconds
        chunk.to_pandas()                     # or chunk.to_polars()
        if done:
            break

    window = band.latest("eeg", seconds=8)    # most recent 8 s, for live models and plots
    print(band.health())

session = openfrenz.export(band.recording, "sessions/")   # same result as `frenz export`
```

## API

### Module functions

| Function | Returns | Notes |
|---|---|---|
| `scan(timeout=8.0)` | `list[Discovered]` (`name`, `rssi`) | |
| `connect(name, *, record=True, reconnect=300.0, scan_timeout=30.0, command_timeout=2.0, buffer_seconds=180.0, start=True)` | `Band` | `record=True` (default) writes `recordings/NAME-UNIXTIME.sqlite`, as the command-line tool does; a file or directory path chooses the location; `record=False` records nothing. `reconnect=None` turns reconnection off; `start=False` sends no start sequence |
| `replay(recording, *, speed=1.0, record=None, buffer_seconds=180.0)` | `Band` | Plays a recording file back through the same path live data takes; no band or Bluetooth needed. Commands raise `Disconnected`. For tests, demos and development. Times, segments and gap counts depend on when packets arrive, so they are only faithful at `speed=1` |
| `synth(path, seconds=60)` | `None` | Writes a synthetic recording, as `frenz synth` does |
| `export(recording, out_dir="sessions")` | `Session` | Runs the Rust exporter (the same one as `frenz export`, byte-identical output) and loads the result from `out_dir/<recording name>/` |
| `summary(recording)` | `dict` | Same content as `frenz summary`: `seconds`, `samples` and `rates_hz` per stream, and `health` counters |
| `COMMANDS` | `dict` | Command name → argument names |
| `load_session`, `find_sessions`, `WindowDataset` | unchanged | |

### `Band`

Use it as a context manager; leaving the block stops the streams, closes the recording and
disconnects, also after an exception or Ctrl-C.

| Member | What it does |
|---|---|
| `info` | name, firmware version, IMU calibration report |
| `stream(name)` | Iterator of `Chunk` for one stream; each item holds the samples that arrived since the last one. Ends when a replay finishes or the band is closed; raises `Disconnected` if the link is lost for good |
| `read(name, timeout=None)` | One `Chunk` with everything that arrived since the last `read` of that stream, or `None` on timeout. Raises `Disconnected` once the connection has ended and everything was read |
| `latest(name, seconds)` | The most recent `seconds` of a stream as one `Chunk` (does not consume). For `eeg`, `accel`, `gyro` and `ppg` this is a sample count (`seconds` × nominal rate), so the shape is fixed |
| `events()` | Iterator of `Event` as they happen; ends when the connection is over. Meant for its own thread |
| `read_event(timeout=None)` | One `Event`, or `None` on timeout (`timeout=0` only checks, for use inside a data loop). Raises `Disconnected` once the connection has ended and every event was read |
| `health()` | Counters: notifications, decode errors, sequence gaps, frames lost, reconnects, last battery and contact state, and per stream under `"streams"`: samples received, buffered and dropped by a slow reader, current segment, and the live clock's fitted rate and jitter |
| `battery()` | Battery percentage |
| `mark(label)` | Writes a marker into the recording with the current time |
| `send(command, ...)` / `request(command, ...)` | Send a command by name, e.g. `band.send("set_prompt_volume", 0)` or `band.request("get_serial_number").value`. `request` waits for and returns a `Response` (`opcode`, `status`, `ok`, `payload`, `value`). Names and arguments are checked before anything is sent |
| `recording` | Path of the recording file, or `None` with `record=False` |
| `close()` | Same as leaving the `with` block; returns the final `health()` |

Streams: `eeg`, `accel`, `gyro`, `ppg`, `heart_rate`, `spo2`, `battery`, `contact`. The first four
have the same names and channels as `openfrenz.SIGNALS` and the exported tables (`heart_rate` and
`spo2` are exported together as `vitals`). Each stream has one read position, so read a given
stream from one place.

### `Chunk`

| Field | Shape and type | Meaning |
|---|---|---|
| `data` | `(channels, n)` int32 | Raw device counts, same channel order and units as the exported session |
| `t` | `(n,)` float64 | Estimated acquisition time, Unix seconds (see [Live timing](#live-timing)) |
| `arrival` | `(n,)` float64 | When the packet carrying each sample reached the computer |
| `device_time` | `(n,)` int64 | The band's own clock for the packet (whole seconds) |
| `segment` | `(n,)` int64 | Uninterrupted stretch the sample belongs to, decided by the same code as in exports |
| `duplicate` | `(n,)` bool | The packet repeated the previous packet's samples |
| `held` | `(n,)` bool | Every sample in the packet was identical (frozen sensor output) |
| `channels` | tuple of str | Channel names |
| `stream` | str | Stream name |
| `dropped` | int | Samples that left the buffer unread before this chunk (see below) |
| `sampled` | bool | False for `heart_rate`, `spo2`, `battery` and `contact`, which have no steady rate: their `t` is the arrival time and `segment`, `duplicate`, `held` carry no information |

The arrays are NumPy. Repeated and frozen packets are delivered and flagged, not removed.
`chunk.to_pandas()` and `chunk.to_polars()` return a table with one row per sample, using the
column names of the exported Parquet table for that stream (`t`, `segment`, `device_time`,
`arrival`, one column per channel, `duplicate`, `held`), so live and exported data can share code.
Exports have a few more columns (`packet`, `sample`, and for EEG `sequence` and `flag`).
pandas is already a dependency; polars is optional (`pip install "openfrenz[polars]"`) and
`to_polars()` says so if it is missing.

The PPG channels are `tag2` and `tag3`, and EEG has its four active channels. The columns left out
are zero on the hardware tested; `health()` counts any sample where one is not
(`inactive_nonzero`), and the recording holds them regardless.

### `Event`

`kind`, `t` (Unix seconds) and `detail` (a dict, empty for most kinds).

| Kind | Meaning | Detail |
|---|---|---|
| `sequence_gap` | EEG frames are missing | `expected`, `received`, `frames_lost` |
| `counter_restart` | The frame counter restarted after a stream start (not data loss) | |
| `off_head` / `on_head` | Electrode contact lost or restored | |
| `power_off_warning` | The band is about to power itself off | |
| `disconnected`, `reconnecting`, `reconnected`, `reconnect_failed` | Link state | `attempt` for `reconnecting` |
| `decode_error` | A notification could not be decoded (it is still in the recording) | `stream`, `error` |
| `lagged` | `read()` was too slow and samples left the buffer | `stream`, `dropped` |
| `events_missed` | The buffers themselves fell behind the band's event queue | `count` |

Up to 4096 unread events are kept; older ones are dropped and counted in `health()`
(`events_dropped`).

### Commands

`get_firmware_version`, `get_battery`, `get_serial_number`, `get_device_name`, `get_device_time`,
`get_device_id`, `get_imu_calibration`, `time_sync([unix_seconds])` (the computer's time if
omitted), `heart_rate_periodic(enable)`, `stream_control(mask, enable)`,
`set_prompt_volume(volume)`, `on_connected`, `start_session`, `end_session`. These are the commands the Rust
client exposes; an unknown name or wrong arguments raise `ValueError` / `TypeError` without
sending anything.

### Errors

All derive from `openfrenz.FrenzError`: `BandNotFound`, `BluetoothError` and its subclass
`BluetoothUnavailable` (no adapter, or the operating system refused access; the message says how
to grant it), `CommandTimeout`,
`CommandFailed`, `Disconnected`, `RecordingError`.

## How it behaves

### Threads

The Bluetooth connection, decoding and recording run on Rust threads that never wait for the
Python interpreter. Python calls only move data out of a buffer. A blocking call such as
`read()` releases the interpreter lock while it waits and returns promptly on Ctrl-C.

### A slow reader loses nothing permanent

Each stream has a ring buffer of `buffer_seconds`. If Python does not read for longer than that,
the oldest samples leave the buffer. This is never silent: the next chunk says how many samples
were skipped in `dropped`, a `lagged` event says the same, and `health()` counts them per stream. `segment` is not changed by this, so it keeps meaning the same as in exports. Unless
recording was turned off, the recording still contains every packet, because it is written by Rust
independently of what Python reads.

### Disconnects

The client reconnects automatically for `reconnect` seconds. Python sees `disconnected`,
`reconnecting` and `reconnected` events, and data after a reconnect is in a new `segment`.
`stream()` keeps waiting across a reconnect and ends with `Disconnected` only when reconnection
is abandoned.

### What cannot be hardened in-process

If the Python process itself exits (for example a notebook kernel restart), the connection goes
with it. The recording up to that moment is intact and can be exported.

### Live timing

Exports estimate each sample's acquisition time by fitting the band's sample counter against
arrival times over a whole segment. Live code cannot wait for the whole segment, so `Chunk.t` uses
the same method on the packets seen so far in the segment (the most recent 1024). Times already
handed out are never revised, and `t` never runs backwards within a segment: when the estimate
moves earlier, the correction is taken up over the following samples.

Measured by replaying two recordings that contain reconnects and comparing with their exports:
once a segment is a few seconds old the two agree to about 5 to 30 ms. In the first seconds of a
segment the live estimate can be a few hundred milliseconds off (up to 0.9 s seen right after a
reconnect), because the first packets on a new connection arrive late and nothing yet shows it.
Analyses that need the best timing should use the exported session; `Chunk.t` is for live display
and live models. The live fit's rate and jitter are reported per stream in `health()`.

## Packaging

- One repository. A new crate `crates/frenz-py` (not published to crates.io) holds the bindings
  and is built with maturin into the `openfrenz` wheel as `openfrenz._native`; the existing Python
  modules stay as they are.
- Wheels for macOS (Apple silicon, Intel), Linux (x86-64, ARM64) and Windows (x86-64), built
  against Python's stable ABI so one wheel per platform serves Python 3.10 and later.
- A source distribution is published as well; installing from it needs Rust.
- The wheel also carries the `frenz` command, so `pip install openfrenz` is enough to scan, record
  and export from a terminal.
- Releases are built and uploaded by GitHub Actions with PyPI trusted publishing. Rust crates and
  the Python package share one version number.
- To verify: bundling the D-Bus library in Linux wheels.

## Testing

- Tests run against a replayed synthetic recording, which goes through the same decoder and event
  path a real band uses, so CI needs no hardware and no Bluetooth.
- Covered so far: chunk shapes and types, every sample delivered, the slow-reader path (what was
  missed is reported and counted, and the recording is still complete), `latest`, timeouts, the
  end of a stream, the live clock, table output, a counter restart after a recorded start
  command, gap and lag events, command argument checking, `summary`, and `export()` agreeing with
  what was streamed. Not covered: reconnect events and command responses, which need a link.
- The live Bluetooth path is tested by hand on each operating system before a release.

## Steps

1. **Done.** A replay source in the Rust client (recorded packets and commands through the real
   decoder and event path). `frenz-py` crate with `scan`, `connect`, `info`, `health`, `battery`,
   `mark`, recording, `close`; maturin build.
2. **Done.** Streams: ring buffers, `read`, `stream`, `latest`, `Chunk` with pandas and polars
   output, lag reporting, live timing. Segment rules shared with the exporter.
3. **Done.** Events, `send` / `request`, `export`, `summary` (the summary logic moved from the
   command-line tool into the export library so both use it).
4. Wheels in CI for all platforms; release to PyPI.
5. Documentation: a Python guide next to the CLI guide.

## Later

- `async` versions of the blocking calls for use inside asyncio programs.
- An optional helper process that owns the connection, so that it survives a Python restart and
  several programs can read one band.

## Decisions

- Recording is on by default. `connect()` prints the recording path once so the file is not a
  surprise, and `record=False` turns it off.
- Chunks are NumPy, with `to_pandas()` and `to_polars()`.
- Blocking calls come first; `async` versions are listed under [Later](#later).
