"""Models described by their files: a ``config.json`` that wires standard blocks together."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from . import _native


class Pipeline:
    """A model made of standard blocks, loaded from a folder or a Hugging Face repository.

    ``source`` is a folder holding ``config.json`` (and ``model.safetensors`` if any block needs
    stored numbers), or a repository written ``owner/name`` or ``owner/name@revision``. Private
    repositories use the token in ``HF_TOKEN`` or the Hugging Face command-line login.

        model = openfrenz.Pipeline("my-model")
        scores = model.run("recordings/session.sqlite")        # a DataFrame, one row per step
        with openfrenz.connect("FRENZ-NAME") as band:
            for out in model.stream(band):                      # one dict per step, as it happens
                print(out)
    """

    def __init__(self, source: str | Path):
        self._native = _native.Pipeline(str(source))

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def version(self) -> str:
        return self._native.version

    @property
    def folder(self) -> Path:
        """Where the model's files are on this computer."""
        return Path(self._native.folder)

    @property
    def inputs(self) -> dict[str, int]:
        """Samples per step of each stream the model reads; the first sets the steps."""
        return dict(self._native.inputs)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Names of the values each step returns."""
        return tuple(self._native.outputs)

    def reset(self) -> None:
        """Forgets everything carried from earlier steps. Call it wherever data is missing."""
        self._native.reset()

    def step(self, data: dict[str, np.ndarray]) -> dict[str, float] | None:
        """One step from ``{stream: (channels, samples) array}``; ``None`` while blocks warm up."""
        blocks = []
        for stream, samples in self.inputs.items():
            x = np.ascontiguousarray(data[stream], dtype="<f8")
            if x.ndim != 2 or x.shape[1] != samples:
                raise ValueError(f"{stream} must be (channels, {samples}), got {x.shape}")
            blocks.append((stream, x.tobytes(), x.shape[0]))
        values = self._native.step(blocks)
        return None if values is None else dict(zip(self.outputs, values))

    def run(self, recording: str | Path) -> pd.DataFrame:
        """Scores a ``.sqlite`` recording: ``t`` (time of each step's last sample) and the outputs.

        Runs a fresh copy of the model, restarted wherever the recording has a hole. Steps that
        had to be skipped are counted in ``result.attrs["missed"]``, by the stream that was
        missing from each.
        """
        t, values, missed = self._native.run(str(recording))
        t = np.frombuffer(t, dtype="<f8")
        table = np.frombuffer(values, dtype="<f8").reshape(len(t), len(self.outputs))
        frame = pd.DataFrame({"t": t, **{name: table[:, k] for k, name in enumerate(self.outputs)}})
        frame.attrs["missed"] = dict(missed)
        return frame

    def stream(self, band) -> ModelStream:
        """Runs a fresh copy of the model on ``band`` from now on; iterate over the result.

        Each item is ``{"t": ..., output: ...}`` for one step, until the band ends. Pauses and
        lost data are recognised by when packets arrive, so replay a recording at its recorded
        pace to see what a live run would give, and use ``run`` to score it quickly.
        """
        return ModelStream(self._native.stream(band), self.outputs)


class ModelStream:
    """A model running on a band: an iterator of one dict per step."""

    def __init__(self, native, names):
        self._native, self._names = native, names

    def __iter__(self) -> Iterator[dict[str, float]]:
        return self

    def __next__(self) -> dict[str, float]:
        item = self._native.__next__()
        if item is None:
            raise StopIteration
        return {"t": item[0], **dict(zip(self._names, item[1]))}

    def read(self, timeout: float) -> dict[str, float] | None:
        """The next step, or ``None`` if none completed within ``timeout`` seconds.

        A model that stays silent is usually waiting for a stream it requires; ``missed`` says
        which. Raises ``StopIteration`` once the band has ended.
        """
        ended, item = self._native.poll(timeout)
        if ended:
            raise StopIteration
        return None if item is None else {"t": item[0], **dict(zip(self._names, item[1]))}

    @property
    def missed(self) -> dict[str, int]:
        """Steps skipped so far, counted by the stream that was missing from each."""
        return dict(self._native.missed)
