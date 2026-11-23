# openfrenz

Unofficial Python package for the FRENZ brainband: connect over Bluetooth LE, read every stream
live, record losslessly, export time-aligned data, and load it for analysis. It is the
[openfrenz](https://github.com/ljchang/openfrenz) Rust client compiled into a Python module; no
Rust and no other software is needed.

```sh
pip install openfrenz
```

```python
import openfrenz

with openfrenz.connect("FRENZA12") as band:       # records to recordings/ by default
    for chunk in band.stream("eeg"):
        chunk.data                                # (4, n) raw counts, NumPy
        chunk.to_pandas()                         # or chunk.to_polars()
        break

session = openfrenz.export(band.recording)        # one Parquet file per stream
t, eeg, segment = session.signal("eeg")
```

The `frenz` command (`scan`, `record`, `summary`, `export`, `synth`) is installed with it.

- [Python guide](https://github.com/ljchang/openfrenz/blob/main/docs/PYTHON.md)
- [Command-line guide](https://github.com/ljchang/openfrenz/blob/main/docs/CLI.md)
- [Hardware notes](https://github.com/ljchang/openfrenz/blob/main/docs/HARDWARE.md)

Not affiliated with or endorsed by Earable. "FRENZ" is used only to describe compatibility.
