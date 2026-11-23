"""Chunks of live samples returned by ``Band.read()``, ``Band.stream()`` and ``Band.latest()``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Chunk:
    """Consecutive samples of one stream.

    ``data`` has shape ``(channels, n)`` and holds raw device counts in the same channel order
    and units as an exported session. The other arrays have one entry per sample.
    """

    #: Stream name (``"eeg"``, ``"accel"``, ...).
    stream: str
    #: Channel names, one per row of ``data``.
    channels: tuple[str, ...]
    #: ``(channels, n)`` int32, raw counts.
    data: np.ndarray
    #: Estimated acquisition time, Unix seconds. Never decreases within a segment.
    t: np.ndarray
    #: When the packet carrying each sample reached the computer, Unix seconds.
    arrival: np.ndarray
    #: The band's own clock for the packet, whole seconds.
    device_time: np.ndarray
    #: Uninterrupted stretch each sample belongs to, numbered as in exports.
    segment: np.ndarray
    #: Position of each sample within its segment. It counts samples that were lost, so a step
    #: larger than 1 between neighbours means data is missing there.
    index: np.ndarray
    #: True where the packet repeated the previous packet's samples.
    duplicate: np.ndarray
    #: True where every sample of the packet was identical (frozen sensor output).
    held: np.ndarray
    #: Samples that left the buffer unread before this chunk, because the reader was too slow.
    #: They are still in the recording.
    dropped: int = 0
    #: False for heart rate, SpO2, battery and contact, which have no steady rate: their ``t``
    #: is the arrival time, and ``segment``, ``index``, ``duplicate`` and ``held`` carry no information.
    sampled: bool = True

    @classmethod
    def _from_buffers(
        cls, stream, channels, sampled, data, t, arrival, device_time, segment, index, duplicate, held, dropped
    ) -> Chunk:
        n = len(t) // 8
        return cls(
            stream=stream,
            channels=tuple(channels),
            data=np.frombuffer(data, dtype=np.int32).reshape(len(channels), n),
            t=np.frombuffer(t, dtype=np.float64),
            arrival=np.frombuffer(arrival, dtype=np.float64),
            device_time=np.frombuffer(device_time, dtype=np.int64),
            segment=np.frombuffer(segment, dtype=np.int64),
            index=np.frombuffer(index, dtype=np.int64),
            duplicate=np.frombuffer(duplicate, dtype=np.bool_),
            held=np.frombuffer(held, dtype=np.bool_),
            dropped=dropped,
            sampled=sampled,
        )

    def __len__(self) -> int:
        return len(self.t)

    def __repr__(self) -> str:
        span = f", t={self.t[0]:.3f}..{self.t[-1]:.3f}" if len(self) else ""
        lost = f", dropped={self.dropped}" if self.dropped else ""
        return f"Chunk({self.stream!r}, {len(self.channels)} channels x {len(self)} samples{span}{lost})"

    def _columns(self) -> dict[str, np.ndarray]:
        """Columns named and ordered as in the exported table for this stream."""
        values = dict(zip(self.channels, self.data))
        if not self.sampled:
            return {"t": self.t, "device_time": self.device_time, **values}
        return {
            "t": self.t,
            "segment": self.segment,
            "device_time": self.device_time,
            "arrival": self.arrival,
            **values,
            "duplicate": self.duplicate,
            "held": self.held,
        }

    def to_pandas(self):
        """One row per sample, with the column names an exported session uses."""
        import pandas as pd

        return pd.DataFrame(self._columns())

    def to_polars(self):
        """One row per sample as a polars DataFrame (needs ``pip install "openfrenz[polars]"``)."""
        try:
            import polars as pl
        except ImportError as error:
            raise ImportError('to_polars() needs polars: pip install "openfrenz[polars]"') from error
        return pl.DataFrame(self._columns())
