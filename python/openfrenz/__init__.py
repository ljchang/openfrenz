"""openfrenz: connect to, stream from and record the FRENZ brainband, and load exported sessions.

Live: ``scan``, ``connect`` and ``replay`` return a ``Band`` whose streams are read as NumPy
``Chunk`` objects. Offline: ``export`` turns a recording into a session, a folder with one Parquet
file per stream plus ``session.json`` (the ``frenz export`` command writes the same thing). Every
sample has ``t``, its estimated acquisition time in Unix seconds on one host clock, so streams are
aligned. Values are raw device units; see ``session.meta``.
"""

from ._native import (
    COMMANDS,
    Band,
    BandInfo,
    BandNotFound,
    BluetoothError,
    BluetoothUnavailable,
    CommandFailed,
    CommandTimeout,
    Disconnected,
    Discovered,
    Event,
    FrenzError,
    RecordingError,
    Response,
    connect,
    replay,
    scan,
    summary,
    synth,
)
from .chunk import Chunk
from .dataset import WindowDataset
from .session import SIGNALS, Session, export, find_sessions, load_session

__all__ = [
    "Band",
    "BandInfo",
    "BandNotFound",
    "BluetoothError",
    "BluetoothUnavailable",
    "COMMANDS",
    "Chunk",
    "CommandFailed",
    "CommandTimeout",
    "Disconnected",
    "Discovered",
    "Event",
    "FrenzError",
    "RecordingError",
    "Response",
    "SIGNALS",
    "Session",
    "WindowDataset",
    "connect",
    "export",
    "find_sessions",
    "load_session",
    "replay",
    "scan",
    "summary",
    "synth",
]
__version__ = "0.3.0"
