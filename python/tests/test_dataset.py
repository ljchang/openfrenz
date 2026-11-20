"""Tests on synthetic sessions, plus real exported sessions when OPENFRENZ_SESSIONS is set."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openfrenz import WindowDataset, find_sessions, load_session


def write_session(root: Path, seconds=30.0, gap=None, duplicate_block=None) -> Path:
    """A synthetic session: EEG 125 Hz, accel 50 Hz; optional gap (t0, t1) and duplicated block."""
    root.mkdir(parents=True)
    tables = {}
    for name, rate, cols in [("eeg", 125.0, ["forehead_right", "comb_left", "forehead_left", "comb_right"]),
                             ("accel", 50.0, ["x", "y", "z"])]:
        n = int(seconds * rate)
        t = 1_800_000_000.0 + np.arange(n) / rate
        segment = np.zeros(n, dtype=np.int64)
        keep = np.ones(n, dtype=bool)
        if gap:
            keep &= ~((t >= gap[0] + t[0]) & (t < gap[1] + t[0]))
            segment[t >= gap[1] + t[0]] = 1
        df = pd.DataFrame({"t": t[keep], "segment": segment[keep]})
        for k, c in enumerate(cols):
            df[c] = np.sin(2 * np.pi * (k + 1) * df["t"].to_numpy()).astype(np.int32 if name == "eeg" else np.int16)
        df["duplicate"] = False
        if duplicate_block and name == "eeg":
            a, b = duplicate_block
            df.loc[(df["t"] >= t[0] + a) & (df["t"] < t[0] + b), "duplicate"] = True
        df.to_parquet(root / f"{name}.parquet")
        tables[name] = {"file": f"{name}.parquet", "rows": len(df)}
    (root / "session.json").write_text(json.dumps({"format": "openfrenz-session/1", "tables": tables}))
    return root


def test_windows_shapes_and_alignment(tmp_path):
    s = write_session(tmp_path / "a", seconds=30)
    ds = WindowDataset([s], window_s=10, hop_s=5, streams=("eeg", "accel"))
    assert len(ds) == 5  # starts at 0, 5, 10, 15, 20 s
    item = ds[0]
    assert item["eeg"].shape == (4, 1250) and item["accel"].shape == (3, 500)
    assert item["eeg"].dtype == np.float32
    assert item["mask"] == {"eeg": True, "accel": True}


def test_gaps_and_duplicates_exclude_windows(tmp_path):
    s = write_session(tmp_path / "b", seconds=40, gap=(12.0, 13.0))
    ds = WindowDataset([s], window_s=10, hop_s=1)
    assert all(not (ref.t0 - 1.8e9 < 13.0 and ref.t0 - 1.8e9 + 10 > 12.0) for ref in ds.windows)
    assert ds.skipped > 0
    d = write_session(tmp_path / "c", seconds=30, duplicate_block=(10.0, 11.0))
    full = WindowDataset([write_session(tmp_path / "d", seconds=30)], window_s=10, hop_s=1)
    dup = WindowDataset([d], window_s=10, hop_s=1)
    assert len(dup) < len(full)  # duplicated samples are removed, leaving a gap


def test_optional_streams_are_masked(tmp_path):
    s = write_session(tmp_path / "e", seconds=20)
    ds = WindowDataset([s], window_s=10, hop_s=10, streams=("eeg", "ppg"), require=("eeg",))
    assert len(ds) >= 1
    assert ds[0]["mask"]["ppg"] is False and ds[0]["ppg"].shape == (2, 250)


SESSIONS = os.environ.get("OPENFRENZ_SESSIONS")


@pytest.mark.skipif(not SESSIONS, reason="set OPENFRENZ_SESSIONS to a folder of exported sessions")
def test_real_sessions_load_and_align():
    paths = find_sessions(SESSIONS)
    assert paths
    for p in paths:
        s = load_session(p)
        if "eeg" not in s:
            continue
        t, x, seg = s.signal("eeg")
        assert np.all(np.diff(t)[np.diff(seg) == 0] > 0), f"{p.name}: time must increase within a segment"
    ds = WindowDataset(paths, window_s=10, hop_s=5, streams=("eeg", "accel", "ppg"), require=("eeg",))
    assert len(ds) > 0
    assert ds[0]["eeg"].shape == (4, 1250)
