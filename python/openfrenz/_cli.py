"""The ``frenz`` command installed with the Python package (the same code as the Rust binary)."""

from __future__ import annotations

import signal
import sys


def main() -> None:
    from . import _native

    # The command handles Ctrl-C itself (it stops the streams and closes the recording), so
    # Python must not also raise KeyboardInterrupt for the same key press.
    signal.signal(signal.SIGINT, lambda *_: None)
    sys.exit(_native.cli(["frenz", *sys.argv[1:]]))
