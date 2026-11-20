# openfrenz (Python)

Load session folders written by `frenz export` and build aligned, fixed-length multimodal windows.

```python
from openfrenz import load_session, WindowDataset, find_sessions

s = load_session("sessions/FRENZA12-1800000000")
t, eeg, segment = s.signal("eeg")          # eeg: (4, n) raw counts
ds = WindowDataset(find_sessions("sessions"), window_s=10, hop_s=5,
                   streams=("eeg", "accel", "gyro", "ppg"), require=("eeg",), min_contact=3)
item = ds[0]   # {"eeg": (4, 1250), "accel": (3, 500), "gyro": (3, 500), "ppg": (2, 250), "mask": {...}}
```

See the [CLI guide](../docs/CLI.md) for the session format.
