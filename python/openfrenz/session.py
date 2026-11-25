"""Loading openfrenz session datasets (folders written by `frenz export`)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT = "openfrenz-session/1"

#: Stream name -> (nominal rate in Hz, signal columns).
SIGNALS: dict[str, tuple[float, tuple[str, ...]]] = {
    "eeg": (125.0, ("forehead_right", "comb_left", "forehead_left", "comb_right")),
    "accel": (50.0, ("x", "y", "z")),
    "gyro": (50.0, ("x", "y", "z")),
    # tag1 is always zero on verified hardware.
    "ppg": (25.0, ("tag2", "tag3")),
}


@dataclass
class Session:
    """One recording session: metadata plus one DataFrame per stream (sorted by time ``t``)."""

    path: Path
    meta: dict
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    def __getitem__(self, stream: str) -> pd.DataFrame:
        return self.tables[stream]

    def __contains__(self, stream: str) -> bool:
        return stream in self.tables

    def span(self, streams=("eeg",)) -> tuple[float, float]:
        """Time range (Unix s) covered by all of ``streams``."""
        present = [self.tables[s] for s in streams if s in self.tables and len(self.tables[s])]
        if not present:
            return (np.nan, np.nan)
        return (max(df["t"].iloc[0] for df in present), min(df["t"].iloc[-1] for df in present))

    def slice(self, t0: float, t1: float) -> dict[str, pd.DataFrame]:
        """All streams restricted to ``t0 <= t < t1``."""
        out = {}
        for name, df in self.tables.items():
            t = df["t"].to_numpy()
            out[name] = df.iloc[np.searchsorted(t, t0) : np.searchsorted(t, t1)]
        return out

    def signal(self, stream: str, include_duplicates: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(t, values[channels, n], segment)`` for a sampled stream. Rows flagged ``duplicate``
        (a repeated block) or ``held`` (frozen sensor output) are removed unless asked for."""
        df = self.tables[stream]
        if not include_duplicates:
            bad = np.zeros(len(df), dtype=bool)
            for flag in ("duplicate", "held"):
                if flag in df:
                    bad |= df[flag].to_numpy()
            df = df[~bad]
        cols = SIGNALS[stream][1]
        seg = df["segment"].to_numpy() if "segment" in df else np.zeros(len(df), dtype=np.int64)
        return df["t"].to_numpy(), df[list(cols)].to_numpy(dtype=np.float64).T, seg


def load_session(path: str | Path) -> Session:
    """Load a session folder written by ``frenz export``."""
    path = Path(path)
    meta = json.loads((path / "session.json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported format {meta.get('format')!r}")
    tables = {}
    for name, info in meta.get("tables", {}).items():
        df = pd.read_parquet(path / info["file"])
        if "t" in df and not df["t"].is_monotonic_increasing:
            df = df.sort_values("t", kind="stable").reset_index(drop=True)
        tables[name] = df
    return Session(path=path, meta=meta, tables=tables)


def find_sessions(root: str | Path) -> list[Path]:
    """All session folders (containing ``session.json``) under ``root``."""
    return sorted(p.parent for p in Path(root).rglob("session.json"))


def export(recording: str | Path, out_dir: str | Path = "sessions", models: Sequence[str | Path] = ()) -> Session:
    """Export a recording (``.sqlite``) to ``out_dir/<recording name>/`` and load the result.

    The same exporter as ``frenz export``: one Parquet file per stream plus ``session.json``.
    Files already in that folder with the same names are replaced.

    ``models`` are run over the recording and each becomes a table named after the model, with
    ``t`` and one column per output: names of models that come with openfrenz
    (``openfrenz.MODELS``), folders, or Hugging Face repositories.

        session = openfrenz.export("recordings/session.sqlite", models=["eog", "heart"])
        session["eog"]                                   # a DataFrame, one row per step
        session.meta["tables"]["eog"]["model"]           # version and fingerprint it came from
    """
    from ._native import export_recording

    return load_session(export_recording(Path(recording), Path(out_dir), [str(m) for m in models]))
