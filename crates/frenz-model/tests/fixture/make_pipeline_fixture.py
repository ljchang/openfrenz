"""Writes the pipeline test fixture: a model folder and the outputs it must give.

The outputs are computed here with NumPy, SciPy and PyTorch, independently of the Rust code that
is tested against them. Needs numpy, scipy, torch and safetensors:

    python crates/frenz-model/tests/fixture/make_pipeline_fixture.py crates/frenz-model/tests/fixture/pipeline
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sps
import torch
from safetensors.torch import save_file

OUT = Path(sys.argv[1])
STEPS, EEG_RATE, ACCEL_RATE, EEG_N, ACCEL_N = 8, 125.0, 50.0, 250, 100
BANDS, TOTAL = [(4, 8), (8, 12), (12, 30)], (1, 45)

CONFIG = {
    "openfrenz_model": "pipeline",
    "name": "fixture",
    "version": "1",
    "step": {"stream": "eeg", "samples": EEG_N},
    "inputs": {"accel": {"samples": ACCEL_N}},
    "nodes": [
        {"name": "clean", "op": "filter", "in": "eeg", "start": "settled", "steps": [
            {"type": "dc_block", "pole": 0.99}, {"type": "notch", "hz": 60, "q": 12},
            {"type": "bandpass", "low": 0.5, "high": 45, "order": 2}]},
        {"name": "power", "op": "band_power", "in": "clean", "bands": BANDS, "relative_to": TOTAL},
        {"name": "beta", "op": "envelope", "in": "clean", "band": [20, 30]},
        {"name": "coh", "op": "coherence", "in": "clean", "pairs": [["forehead_left", "forehead_right"]],
         "bands": [[8, 12]], "segment": 125, "average_steps": 3},
        {"name": "pac", "op": "coupling", "in": "clean", "phase_band": [4, 8], "amplitude_band": [30, 45],
         "trim_s": 0.2, "average_steps": 3},
        {"name": "move", "op": "summary", "in": "accel", "stats": ["sd", "mean"]},
        {"name": "x", "op": "concat", "in": ["power", "beta", "coh", "pac", "move"]},
        {"name": "z", "op": "normalize", "in": "x", "weights": "norm"},
        {"name": "recent", "op": "history", "in": "z", "steps": 3},
        {"name": "net", "op": "network", "in": "recent", "names": ["a", "b"], "layers": [
            {"type": "lstm", "weights": "rnn"}, {"type": "last"},
            {"type": "dense", "weights": "head.0", "activation": "elu"},
            {"type": "dense", "weights": "head.2"}, {"type": "sigmoid"}]},
    ],
    "outputs": ["net"],
}


def eeg(step):
    """Four channels: a 10 Hz rhythm, a 6 Hz rhythm that waxes and wanes, and 40 Hz bursts riding on it."""
    t = (step * EEG_N + np.arange(EEG_N)) / EEG_RATE
    w = 2 * np.pi * t
    return np.stack([2000 * np.sin(10 * w + c) + 800 * np.sin(6 * w) * (1 + 0.5 * np.sin(0.3 * w + c))
                     + 300 * np.sin(40 * w * (1 + 0.01 * c)) * (1 + 0.6 * np.sin(6 * w)) + 5000 + 100 * c for c in range(4)])


def accel(step):
    t = (step * ACCEL_N + np.arange(ACCEL_N)) / ACCEL_RATE
    w = 2 * np.pi * t
    return np.stack([16000.0 * (a == 0) + 50 * np.sin(1.5 * w + a) + 20 * np.sin(7 * w) for a in range(3)])


def band_sos(lo, hi, rate=EEG_RATE):
    return sps.butter(2, [lo, hi], "band", fs=rate, output="sos")


def analytic(x, lo, hi):
    return sps.hilbert(sps.sosfiltfilt(band_sos(lo, hi), x))


def stretches(x, segment=125):
    hop = segment - segment // 2
    return [np.fft.rfft((x[i:i + segment] - x[i:i + segment].mean()) * sps.get_window("hann", segment))
            for i in range(0, len(x) - segment + 1, hop)]


torch.manual_seed(0)
rng = np.random.default_rng(0)
N = 12 + 8 + 1 + 8 + 6
rnn, head = torch.nn.LSTM(N, 8, batch_first=True), torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ELU(), torch.nn.Linear(4, 2))
center, scale = rng.normal(0, 0.2, N), rng.uniform(0.5, 2.0, N)
center[-6:] = [0, 16000, 0, 0, 0, 0]
scale[-6:] = 40

notch = sps.tf2sos(*sps.iirnotch(60, 12, EEG_RATE))
sos = np.vstack([[[1, -1, 0, 1, -0.99, 0]], notch, band_sos(0.5, 45)])
state, cross, sums, rows, features, outputs = None, [], [], [], [], []
for step in range(STEPS):
    raw = eeg(step)
    if state is None:
        state = sps.sosfilt_zi(sos)[:, None, :] * raw[:, :1][None]
    clean, state = sps.sosfilt(sos, raw, zi=state)
    x = []
    for c in range(4):
        f, p = sps.periodogram(clean[c], EEG_RATE, window="hann")
        power = lambda lo, hi: p[(f >= lo) & (f < hi)].sum() * f[1]
        x += [power(*b) / power(*TOTAL) for b in BANDS]
    for c in range(4):
        env = np.abs(analytic(clean[c], 20, 30))
        x += [env.mean(), env.std()]
    a, b = stretches(clean[2]), stretches(clean[0])  # forehead_left, forehead_right
    cross.append((sum(abs(v) ** 2 for v in a), sum(abs(v) ** 2 for v in b), sum(np.conj(u) * v for u, v in zip(a, b))))
    k = (np.fft.rfftfreq(125, 1 / EEG_RATE) >= 8) & (np.fft.rfftfreq(125, 1 / EEG_RATE) < 12)
    xx, yy, xy = (sum(s[i] for s in cross[-3:])[k].sum() for i in range(3))
    x.append(abs(xy) ** 2 / (xx * yy))
    per = []
    for c in range(4):
        ph, am = np.angle(analytic(clean[c], 4, 8))[25:-25], np.abs(analytic(clean[c], 30, 45))[25:-25]
        bins = np.clip(np.floor((ph + np.pi) / (2 * np.pi / 18)).astype(int), 0, 17)
        per.append((np.bincount(bins, am, 18), np.bincount(bins, minlength=18), (am * np.exp(1j * ph)).sum(), (am ** 2).sum(), len(am)))
    sums.append(per)
    for c in range(4):
        amp, cnt, vec, sq, n = (sum(s[c][i] for s in sums[-3:]) for i in range(5))
        m = np.where(cnt > 0, amp / np.maximum(cnt, 1), 0.0)
        q = m[m > 0] / m.sum()
        x += [(np.log(18) + (q * np.log(q)).sum()) / np.log(18), abs(vec) / np.sqrt(n * sq)]
    for axis in accel(step):
        x += [axis.std(), axis.mean()]
    ready = step >= 2  # coherence and coupling have their three steps
    features.append([float(v) for v in x] if ready else None)
    if ready:
        rows.append((np.array(x) - center) / scale)
    if len(rows) >= 3:
        with torch.no_grad():
            h = rnn(torch.tensor(np.array(rows[-3:]), dtype=torch.float32)[None])[0][:, -1]
            outputs.append([float(v) for v in torch.sigmoid(head(h))[0]])
    else:
        outputs.append(None)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "config.json").write_text(json.dumps(CONFIG, indent=1) + "\n")
weights = {f"rnn.{k}": v for k, v in rnn.state_dict().items()} | {f"head.{k}": v for k, v in head.state_dict().items()}
weights |= {"norm.center": torch.tensor(center), "norm.scale": torch.tensor(scale)}
save_file(weights, OUT / "model.safetensors")
(OUT / "expected.json").write_text(json.dumps({"features": features, "outputs": outputs}, indent=1) + "\n")
print(f"{STEPS} steps, {N} features, outputs from step {next(i for i, o in enumerate(outputs) if o)}:", outputs[-1])
