# `openfrenz` Python guide

Connect to a FRENZ brainband, read its streams live, record, and load the results, all from
Python. The package is the Rust client compiled into a Python module, so it behaves exactly like
the [`frenz` command](CLI.md), which is installed with it.

```sh
pip install openfrenz            # or: uv add openfrenz
```

Needs Python 3.10 or newer. Wheels are provided for macOS, Linux and Windows; nothing else has to
be installed. On macOS the application running Python (Terminal, VS Code, …) needs Bluetooth
permission; macOS asks the first time. On Linux, BlueZ must be running.

No band at hand? Every example below also works on a recording: replace
`openfrenz.connect("FRENZA12")` with `openfrenz.replay("demo.sqlite")` after
`openfrenz.synth("demo.sqlite")`.

## Before you connect

- Close the band's phone app, or anything else connected to it: the band accepts one connection.
- Switch the band on and wear it. Electrode contact takes about 10 seconds to settle.
- The band powers itself off about half a minute after it is taken off while streaming.

## Find and connect

```python
import openfrenz

for found in openfrenz.scan():
    print(found.name, found.rssi)

with openfrenz.connect("FRENZA12") as band:
    print(band.info.firmware, band.battery())
```

`connect` starts every stream and records everything the band sends to
`recordings/FRENZA12-<time>.sqlite`. Pass `record="my-file.sqlite"` to choose the file or
`record=False` to switch recording off. Leaving the `with` block stops the streams, closes the
recording and disconnects, also after an error or Ctrl-C.

## Read data

```python
with openfrenz.connect("FRENZA12") as band:
    for chunk in band.stream("eeg"):      # waits for new samples
        chunk.data                        # (4, n) int32, raw counts
        chunk.t                           # (n,) estimated Unix seconds
        chunk.channels                    # ('forehead_right', 'comb_left', ...)
        df = chunk.to_pandas()            # or chunk.to_polars()
        if enough:
            break
```

Streams: `eeg` (125 Hz), `accel` and `gyro` (50 Hz), `ppg` (25 Hz), `heart_rate`, `spo2`,
`battery` and `contact` (0–3, where 3 is good electrode contact).

| Call | Use it for |
|---|---|
| `band.stream(name)` | A loop over everything that arrives |
| `band.read(name, timeout=None)` | One chunk with everything since the last read; `None` on timeout |
| `band.latest(name, seconds)` | The most recent stretch without consuming it: live plots and live models |

Values are raw device counts; see [HARDWARE.md](HARDWARE.md) for what is known about scales.

To stream the band to Lab Streaming Layer, or watch it in a real-time dashboard, without writing code, use
[HyperStudy Bridge](BRIDGE.md): it publishes these streams and the models' outputs over LSL.
Nothing is removed or filled in: a packet the band repeated is delivered and marked in
`chunk.duplicate`, frozen sensor output in `chunk.held`, and a break in the data starts a new
number in `chunk.segment`.

If your code does not read for longer than the buffer (`buffer_seconds`, 3 minutes by default),
the oldest samples leave it. The next chunk reports how many in `chunk.dropped`. The recording
still has all of them.

## Markers, events and health

```python
with openfrenz.connect("FRENZA12") as band:
    band.mark("task_start")                       # stored in the recording with the current time
    for chunk in band.stream("eeg"):
        event = band.read_event(timeout=0)        # check without waiting
        if event is not None:
            print(event.kind, event.detail)       # e.g. 'off_head', 'sequence_gap', 'reconnected'
    print(band.health())                          # counts of packets, gaps, reconnects, ...
```

If the connection drops, the client reconnects by itself for up to `reconnect` seconds (default
300) and the loop above carries on; data after the reconnect is in a new segment.

## Commands

```python
band.request("get_serial_number").value
band.send("set_prompt_volume", 0)
openfrenz.COMMANDS            # every command name and its arguments
```

## After the session

```python
openfrenz.summary(band.recording)                 # duration, sample counts, rates, health
session = openfrenz.export(band.recording)        # writes sessions/<name>/ and loads it
t, eeg, segment = session.signal("eeg")           # the best time estimate for every sample
```

`export` gives the same files as `frenz export`. Exported times are more accurate than live ones,
because they are estimated from the whole recording; use the export for analysis and `chunk.t` for
live display. Build fixed-length training windows from exported sessions with `WindowDataset`:

```python
from openfrenz import WindowDataset, find_sessions

windows = WindowDataset(find_sessions("sessions"), window_s=10, hop_s=5,
                        streams=("eeg", "accel", "gyro", "ppg"), min_contact=3)
windows[0]["eeg"].shape                           # (4, 1250)
```

## Process signals and run models

`openfrenz.dsp` has filters, spectra, wavelets, coherence, coupling, heartbeat detection and
artifact removal, on NumPy arrays:

```python
from openfrenz import dsp

clean = dsp.Filter([{"type": "notch", "hz": 60}, {"type": "bandpass", "low": 0.5, "high": 45}], 125)
freqs, density = dsp.welch(clean(eeg), 125, segment=250)
alpha = dsp.band_power(freqs, density, 8, 12)
```

`openfrenz.Pipeline` runs a model described by its files, from a folder or a Hugging Face
repository, on a recording or on a band as it streams:

```python
model = openfrenz.Pipeline("owner/name")
scores = model.run(band.recording)                 # a DataFrame, one row per step
for out in model.stream(band):                     # live, one dict per step
    print(out)
```

See [Models and signal processing](MODELS.md) for every block and the pipeline format.

## From a terminal

`pip install openfrenz` also installs the `frenz` command (`frenz scan`, `frenz record`,
`frenz summary`, `frenz export`, `frenz synth`); see the [CLI guide](CLI.md).

## Limits

- Tested with one band (firmware `X8.2.2_1.1.19`) on macOS. The Linux and Windows wheels pass the
  tests that need no band but have not been tried with one.
- One connection per band and one reader per stream.
- If the Python process exits (for example a notebook kernel restart), the connection goes with
  it. The recording up to that moment is intact.

The full reference, including timing details, is in [PYTHON_API.md](PYTHON_API.md).
