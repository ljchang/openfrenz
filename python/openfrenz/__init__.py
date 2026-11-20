"""openfrenz: load FRENZ brainband session datasets and build ML-ready windows.

Sessions are folders written by the ``frenz export`` command (Parquet per stream plus
``session.json``). Every sample has ``t``, its estimated acquisition time in Unix seconds on one
host clock, so streams are aligned. Values are raw device units; see ``session.meta``.
"""

from .dataset import WindowDataset
from .session import SIGNALS, Session, find_sessions, load_session

__all__ = ["SIGNALS", "Session", "WindowDataset", "find_sessions", "load_session"]
__version__ = "0.1.0"
