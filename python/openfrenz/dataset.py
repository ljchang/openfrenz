"""Fixed-length, time-aligned multimodal windows.

``WindowDataset`` is a plain sequence (``__len__`` and ``__getitem__``) of NumPy arrays, so it can be
used directly or wrapped by any data-loading framework. Each item holds every requested stream at its native
rate for the same time window, so EEG (125 Hz), motion (50 Hz) and PPG (25 Hz) stay aligned without
resampling. A window is only valid for a stream when it lies inside one uninterrupted segment with no
gap and no duplicated samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .session import SIGNALS, Session, load_session


@dataclass(frozen=True)
class WindowRef:
    session: int
    t0: float


class _StreamIndex:
    """Per-stream arrays and a fast validity check for windows."""

    def __init__(self, session: Session, stream: str):
        rate, _ = SIGNALS[stream]
        self.rate = rate
        self.t, self.values, self.segment = session.signal(stream)
        self.values = self.values.astype(np.float32)

    def take(self, t0: float, n: int) -> np.ndarray | None:
        i = int(np.searchsorted(self.t, t0))
        j = i + n
        if j > len(self.t):
            return None
        tw = self.t[i:j]
        period = 1.0 / self.rate
        if tw[0] - t0 > 1.5 * period:  # window starts in a gap
            return None
        if self.segment[i] != self.segment[j - 1]:
            return None
        if np.any(np.diff(tw) > 1.5 * period):
            return None
        return self.values[:, i:j]


class WindowDataset:
    """Sliding windows over one or more sessions.

    Parameters
    ----------
    sessions:
        Session folders or loaded :class:`Session` objects.
    window_s, hop_s:
        Window length and step in seconds.
    streams:
        Streams to return (any of ``eeg``, ``accel``, ``gyro``, ``ppg``).
    require:
        Streams that must be valid for a window to be included; other streams are returned
        zero-filled with ``mask[stream] = False`` when unavailable.
    min_contact:
        If set, skip windows where the band's contact state (0–3) drops below this value.
    transform:
        Optional callable applied to each item dict.
    """

    def __init__(
        self,
        sessions: Iterable[str | Path | Session],
        window_s: float = 10.0,
        hop_s: float = 5.0,
        streams: Sequence[str] = ("eeg", "accel", "gyro", "ppg"),
        require: Sequence[str] = ("eeg",),
        min_contact: int | None = None,
        transform: Callable[[dict], dict] | None = None,
    ):
        unknown = set(streams) - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown streams {sorted(unknown)}")
        self.sessions = [s if isinstance(s, Session) else load_session(s) for s in sessions]
        self.window_s, self.hop_s = float(window_s), float(hop_s)
        self.streams, self.require = tuple(streams), tuple(require)
        self.min_contact, self.transform = min_contact, transform
        self.lengths = {s: int(round(self.window_s * SIGNALS[s][0])) for s in self.streams}
        self._index = [{s: _StreamIndex(sess, s) for s in self.streams if s in sess} for sess in self.sessions]
        self.windows: list[WindowRef] = []
        self.skipped = 0
        for k, sess in enumerate(self.sessions):
            idx = self._index[k]
            if any(r not in idx for r in self.require):
                continue
            start, end = sess.span(self.require)
            if not np.isfinite(start):
                continue
            contact = sess.tables.get("contact")
            # The span ends at the last sample's time, one period before the data ends.
            last_period = 1.0 / min(SIGNALS[r][0] for r in self.require)
            for t0 in np.arange(start, end - self.window_s + last_period + 1e-4, self.hop_s):
                ok = all(idx[r].take(t0, self.lengths[r]) is not None for r in self.require)
                if ok and self.min_contact is not None and contact is not None and len(contact):
                    c = contact[(contact["t"] >= t0) & (contact["t"] < t0 + self.window_s)]["state"]
                    ok = len(c) > 0 and int(c.min()) >= self.min_contact
                if ok:
                    self.windows.append(WindowRef(k, float(t0)))
                else:
                    self.skipped += 1

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        ref = self.windows[i]
        idx = self._index[ref.session]
        item: dict = {"session": self.sessions[ref.session].name, "t0": ref.t0, "mask": {}}
        for s in self.streams:
            channels = len(SIGNALS[s][1])
            x = idx[s].take(ref.t0, self.lengths[s]) if s in idx else None
            item["mask"][s] = x is not None
            item[s] = x if x is not None else np.zeros((channels, self.lengths[s]), dtype=np.float32)
        return self.transform(item) if self.transform else item
