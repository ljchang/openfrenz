"""Signal-processing blocks: filters, spectra, wavelets, envelopes, coherence, coupling,
heartbeats and artifact reconstruction.

The same code that runs inside pipelines (``openfrenz.Pipeline``) and in the Rust crate
``frenz-model``, so features computed here for training are the ones a deployed model computes.
Arrays are NumPy; a signal is ``(n,)`` for one channel or ``(channels, n)``. Filters and spectra
follow SciPy's numerics (``butter``, ``sosfilt``, ``sosfiltfilt``, ``periodogram``, ``welch``,
``hilbert``, ``coherence``).

    from openfrenz import dsp
    clean = dsp.Filter([{"type": "notch", "hz": 60}, {"type": "bandpass", "low": 0.5, "high": 45}], 125)
    y = clean(x)                                  # keeps its state: call it step after step
    freqs, density = dsp.welch(y, 125, segment=125)
    alpha = dsp.band_power(freqs, density, 8, 12)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import _native

__all__ = [
    "Asr",
    "Coupling",
    "Filter",
    "Heart",
    "analytic",
    "asr_calibrate",
    "band_power",
    "coherence",
    "coupling",
    "heart",
    "morlet",
    "periodogram",
    "welch",
]


def _floats(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype="<f8")


def _array(buffer, shape=None) -> np.ndarray:
    a = np.frombuffer(buffer, dtype="<f8")
    return a if shape is None else a.reshape(shape)


def _each(x, function):
    """Applies ``function`` to one channel, or to each row of ``(channels, n)``, stacking the results."""
    x = _floats(x)
    if x.ndim == 1:
        return function(x)
    if x.ndim != 2:
        raise ValueError(f"a signal is (n,) or (channels, n), not {x.shape}")
    parts = [function(row) for row in x]
    if isinstance(parts[0], tuple):
        return tuple(np.stack(p) for p in zip(*parts))
    return np.stack(parts)


class Filter:
    """A chain of filters that keeps its state, so a signal filtered in pieces equals one filtered whole.

    ``steps`` is one filter or a list, each a dict with ``type`` and its numbers:
    ``lowpass`` (``high``, ``order``), ``highpass`` (``low``, ``order``), ``bandpass`` (``low``,
    ``high``, ``order``), ``notch`` (``hz``, ``q``) or ``dc_block`` (``pole``). Defaults: order 2,
    q 30, pole 0.99. A band-pass of ``order`` has ``2 * order`` poles, as in SciPy.

    ``start="settled"`` begins as if the first sample had been held for ever, which avoids the
    switch-on transient of a signal with an offset; ``"zero"`` begins from silence.
    """

    def __init__(self, steps: dict | Sequence[dict], rate_hz: float, start: str = "zero"):
        if start not in ("zero", "settled"):
            raise ValueError(f"start must be 'zero' or 'settled', not {start!r}")
        steps = [steps] if isinstance(steps, dict) else list(steps)
        self.steps, self.rate_hz, self.start = steps, float(rate_hz), start
        self._native = _native.Filter(json.dumps(steps), float(rate_hz), start == "settled")
        self._channels: int | None = None

    def _run(self, x, method) -> np.ndarray:
        x = _floats(x)
        if x.ndim not in (1, 2):
            raise ValueError(f"a signal is (n,) or (channels, n), not {x.shape}")
        channels = 1 if x.ndim == 1 else x.shape[0]
        return _array(method(x.tobytes(), channels), x.shape).copy()

    def __call__(self, x) -> np.ndarray:
        """Filters the next samples, continuing from the previous call."""
        return self._run(x, self._native.process)

    def filtfilt(self, x) -> np.ndarray:
        """Filters forwards then backwards, for no delay. Needs the whole signal; uses no state."""
        return self._run(x, self._native.filtfilt)

    def reset(self) -> None:
        """Forgets every sample seen."""
        self._native.reset()


def periodogram(x, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies, density)``: power per hertz of the whole signal, mean removed, Hann window."""
    density = _each(x, lambda row: _array(_native.spectrum(row.tobytes(), float(rate_hz))[1]).copy())
    n = np.shape(x)[-1]
    return np.arange(density.shape[-1]) * (rate_hz / n if n else 0.0), density


def welch(x, rate_hz: float, segment: int) -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies, density)`` averaged over half-overlapping stretches of ``segment`` samples."""
    density = _each(x, lambda row: _array(_native.spectrum(row.tobytes(), float(rate_hz), int(segment))[1]).copy())
    size = min(int(segment), np.shape(x)[-1])
    return np.arange(density.shape[-1]) * (rate_hz / size if size else 0.0), density


def band_power(frequencies, density, low_hz: float, high_hz: float) -> np.ndarray:
    """Power from ``low_hz`` (included) to ``high_hz`` (excluded): density times bin width, summed."""
    frequencies, density = np.asarray(frequencies), np.asarray(density)
    width = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0.0
    return density[..., (frequencies >= low_hz) & (frequencies < high_hz)].sum(-1) * width


def analytic(x) -> tuple[np.ndarray, np.ndarray]:
    """``(amplitude, phase)`` of a band-limited signal: its envelope, and its phase in radians."""
    return _each(x, lambda row: tuple(_array(b).copy() for b in _native.analytic(row.tobytes())))


def morlet(x, rate_hz: float, hz: float, cycles: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """``(amplitude, phase)`` at ``hz`` sample by sample, from a Morlet wavelet of ``cycles`` cycles.

    The amplitude is in signal units: a sine of amplitude 3 at ``hz`` reads 3.
    """
    return _each(x, lambda row: tuple(_array(b).copy() for b in _native.morlet(row.tobytes(), float(rate_hz), float(hz), float(cycles))))


def coherence(x, y, rate_hz: float, segment: int) -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies, coherence)`` between two channels, from half-overlapping stretches of ``segment`` samples.

    0 for unrelated signals, 1 for a fixed linear relation. Needs several stretches: a single
    one always reads 1.
    """
    x, y = _floats(x), _floats(y)
    resolution, values = _native.coherence(x.tobytes(), y.tobytes(), float(rate_hz), int(segment))
    values = _array(values).copy()
    return np.arange(len(values)) * resolution, values


@dataclass(frozen=True)
class Coupling:
    """How strongly the size of one rhythm follows the phase of another."""

    #: Tort's modulation index: 0 for none; a few hundredths is already strong.
    modulation_index: float
    #: Length of the amplitude-weighted mean phase vector, between 0 and 1 whatever the amplitude.
    mean_vector_length: float
    #: The phase at which the amplitude is largest, in radians.
    preferred_phase: float


def coupling(phase, amplitude, bins: int = 18) -> Coupling:
    """Phase-amplitude coupling between a phase and an amplitude at the same samples.

    Take the phase of a slow band and the amplitude of a faster one from ``analytic``, and leave
    out the ends of the signal. Both measures need many cycles of the slow rhythm.
    """
    return Coupling(*_native.coupling(_floats(phase).tobytes(), _floats(amplitude).tobytes(), int(bins)))


@dataclass(frozen=True, eq=False)
class Heart:
    """Heartbeats found in a stretch of pulse signal."""

    #: Time of each beat, seconds from the start of the stretch.
    beats: np.ndarray
    #: Beats per minute; NaN when too few trustworthy intervals were found.
    rate_bpm: float
    #: Standard deviation of the intervals between beats, in milliseconds.
    sdnn_ms: float
    #: Root mean square of the change from one interval to the next, in milliseconds.
    rmssd_ms: float
    #: Share of intervals that look like heartbeats; distrust the rest below about 0.8.
    quality: float


def heart(ppg, rate_hz: float) -> Heart:
    """Beats, heart rate and its variability from one pulse channel (pulses pointing up)."""
    found = _native.heart(_floats(ppg).tobytes(), float(rate_hz))
    return Heart(_array(found["beats"]).copy(), found["rate_bpm"], found["sdnn_ms"], found["rmssd_ms"], found["quality"])


def asr_calibrate(reference, rate_hz: float, cutoff: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Learns artifact reconstruction from clean data ``(channels, n)``: ``(mixing, threshold)``.

    Use at least a minute of quiet data, filtered as the data to be cleaned will be. ``cutoff``
    is how many times its usual spread a direction may exceed its usual size before it is
    removed; 20 to 30 is the recommended range, and lower values remove more.
    """
    reference = _floats(reference)
    if reference.ndim != 2:
        raise ValueError("the reference is (channels, n)")
    c = reference.shape[0]
    mixing, threshold = _native.asr_calibrate(reference.tobytes(), c, float(rate_hz), float(cutoff))
    return _array(mixing, (c, c)).copy(), _array(threshold, (c, c)).copy()


class Asr:
    """Artifact subspace reconstruction: removes large, brief disturbances from multi-channel data.

    Directions of the signal that are far larger than they were in the clean data are removed
    and rebuilt from the rest. Keeps its state between calls, so call it step after step.
    """

    def __init__(self, mixing, threshold, max_removed: float = 0.66, window_s: float = 0.5):
        mixing, threshold = _floats(mixing), _floats(threshold)
        if mixing.ndim != 2 or mixing.shape[0] != mixing.shape[1] or threshold.shape != mixing.shape:
            raise ValueError("mixing and threshold are square matrices of one size")
        self.channels = mixing.shape[0]
        self._native = _native.Asr(mixing.tobytes(), threshold.tobytes(), self.channels, float(max_removed), float(window_s))

    def __call__(self, x, rate_hz: float) -> np.ndarray:
        """Cleans the next samples ``(channels, n)``, continuing from the previous call."""
        x = _floats(x)
        if x.ndim != 2 or x.shape[0] != self.channels:
            raise ValueError(f"expected ({self.channels}, n), got {x.shape}")
        return _array(self._native.process(x.tobytes(), self.channels, float(rate_hz)), x.shape).copy()

    def reset(self) -> None:
        """Forgets the previous window."""
        self._native.reset()
