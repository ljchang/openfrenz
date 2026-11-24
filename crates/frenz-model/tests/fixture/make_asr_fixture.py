"""Writes the artifact-reconstruction test fixture: what a NumPy version of the same method gives.

Written from the description in `src/dsp/asr.rs`, independently of that code. Needs numpy:

    python crates/frenz-model/tests/fixture/make_asr_fixture.py crates/frenz-model/tests/fixture/asr
"""
import json
import sys
from pathlib import Path

import numpy as np

RATE, CHANNELS, CUTOFF = 125.0, 4, 20.0
MIX = np.array([[1.0, 0.6, 0.2, 0.1], [0.5, 1.0, 0.3, 0.2], [0.2, 0.4, 1.0, 0.5], [0.1, 0.2, 0.6, 1.0]])


class Lcg:
    """The generator the Rust test uses, so both make the same signals."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.state >> 11) / (1 << 53)


def signal(seed, samples):
    """Four mixed sources of smoothed noise, about 20 units in size."""
    rng = Lcg(seed)
    raw = np.array([[sum(rng.next() for _ in range(4)) - 2.0 for _ in range(CHANNELS)] for _ in range(samples + 2)]).T
    smooth = (raw[:, :-2] + raw[:, 1:-1] + raw[:, 2:]) / 3.0
    return 40.0 * MIX @ smooth


def eigen(m):
    values, vectors = np.linalg.eigh(m)
    return values, vectors


def geometric_median(points):
    centre = points.mean(0)
    for _ in range(1000):
        w = 1.0 / np.maximum(np.linalg.norm(points - centre, axis=1), 1e-12)
        nxt = (points * w[:, None]).sum(0) / w.sum()
        moved = np.linalg.norm(nxt - centre)
        centre = nxt
        if moved <= 1e-12 * np.linalg.norm(centre):
            break
    return centre


def calibrate(x, cutoff):
    window, block = int(0.5 * RATE + 0.5), 10
    blocks = np.array([(x[:, b * block:(b + 1) * block] @ x[:, b * block:(b + 1) * block].T / block).ravel()
                       for b in range(x.shape[1] // block)])
    values, vectors = eigen(geometric_median(blocks).reshape(CHANNELS, CHANNELS))
    mixing = vectors @ np.diag(np.sqrt(np.maximum(values, 0))) @ vectors.T
    _, directions = eigen(mixing)
    activity = directions.T @ x
    hop = max(round(window * 0.34), 1)
    limits = []
    for row in activity:
        sizes = np.array([np.sqrt((row[w * hop:w * hop + window] ** 2).mean()) for w in range((len(row) - window) // hop + 1)])
        middle = np.median(sizes)
        limits.append(middle + cutoff * 1.4826 * np.median(np.abs(sizes - middle)))
    return mixing, np.diag(limits) @ directions.T


def clean(x, mixing, threshold, most, window):
    out, previous, start = np.zeros_like(x), None, 0
    while start < x.shape[1]:
        stop = x.shape[1] if x.shape[1] - start < 2 * window else start + window
        part = x[:, start:stop]
        values, vectors = eigen(part @ part.T / part.shape[1])
        keep = (values < ((threshold @ vectors) ** 2).sum(0)) | (np.arange(CHANNELS) < CHANNELS - most)
        r = np.eye(CHANNELS) if keep.all() else mixing @ np.linalg.pinv(keep[:, None] * (vectors.T @ mixing)) @ vectors.T
        before = r if previous is None else previous
        blend = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, part.shape[1] + 1) / part.shape[1])
        out[:, start:stop] = blend * (r @ part) + (1 - blend) * (before @ part)
        previous, start = r, stop
    return out


reference = signal(1, 7500)
mixing, threshold = calibrate(reference, CUTOFF)
data = signal(2, 1000)
dirty = data.copy()
dirty[1, 400:525] += 600.0 * np.sin(2 * np.pi * 3.0 * np.arange(125) / RATE)  # one second, one channel
cleaned = clean(dirty, mixing, threshold, most=int(0.66 * CHANNELS), window=int(0.5 * RATE + 0.5))
at = [0, 399, 420, 462, 500, 524, 700, 999]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "expected.json").write_text(json.dumps({
    "mixing": mixing.tolist(), "threshold_squared_norms": (threshold ** 2).sum(1).tolist(),
    "at": at, "cleaned": cleaned[:, at].T.tolist(), "data": data[:, at].T.tolist(),
}, indent=1) + "\n")
burst = slice(400, 525)
print("burst rms error on the hit channel: before", np.sqrt(((dirty - data)[1, burst] ** 2).mean()).round(1),
      "after", np.sqrt(((cleaned - data)[1, burst] ** 2).mean()).round(1), "| signal rms", data[1].std().round(1),
      "| max change outside 375..625:", np.abs((cleaned - dirty)[:, np.r_[0:375, 625:1000]]).max())
