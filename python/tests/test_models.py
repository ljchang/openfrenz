"""The signal-processing blocks and pipelines, through the Python bindings."""

import json
from pathlib import Path

import numpy as np
import pytest

import openfrenz
from openfrenz import dsp

FIXTURE = Path(__file__).resolve().parents[2] / "crates/frenz-model/tests/fixture/pipeline"
RATE = 125.0


def signal(n=400):
    k = np.arange(n)
    return np.sin(0.1 * k) + 0.5 * np.sin(1.3 * k + 1) + 0.01 * k


def fixture_step(index):
    """The signals of the pipeline fixture (see ``make_pipeline_fixture.py``)."""
    w = 2 * np.pi * (index * 250 + np.arange(250)) / 125.0
    eeg = np.stack([2000 * np.sin(10 * w + c) + 800 * np.sin(6 * w) * (1 + 0.5 * np.sin(0.3 * w + c))
                    + 300 * np.sin(40 * w * (1 + 0.01 * c)) * (1 + 0.6 * np.sin(6 * w)) + 5000 + 100 * c for c in range(4)])
    w = 2 * np.pi * (index * 100 + np.arange(100)) / 50.0
    accel = np.stack([16000.0 * (a == 0) + 50 * np.sin(1.5 * w + a) + 20 * np.sin(7 * w) for a in range(3)])
    return {"eeg": eeg, "accel": accel}


def test_filter_keeps_its_state_and_handles_channels():
    x = np.stack([signal(), -2 * signal()])
    steps = [{"type": "dc_block"}, {"type": "notch", "hz": 60, "q": 12}, {"type": "bandpass", "low": 0.5, "high": 45}]
    whole = dsp.Filter(steps, RATE, start="settled")(x)
    assert whole.shape == x.shape
    f = dsp.Filter(steps, RATE, start="settled")
    pieces = np.concatenate([f(x[:, i:i + 37]) for i in range(0, 400, 37)], axis=1)
    np.testing.assert_array_equal(pieces, whole)
    np.testing.assert_allclose(whole[1], -2 * whole[0], rtol=1e-12, atol=1e-12)
    f.reset()
    np.testing.assert_array_equal(f(x), whole)
    # Values from SciPy (sosfilt and sosfiltfilt of a 0.5-45 Hz band-pass of order 2).
    band = dsp.Filter({"type": "bandpass", "low": 0.5, "high": 45, "order": 2}, RATE)
    np.testing.assert_allclose(band(signal())[[0, 1, 150]], [0.218732450860704, 0.500972332784362, 0.736663933742422], rtol=1e-9)
    np.testing.assert_allclose(band.filtfilt(signal())[[0, 150]], [0.465252996478147, 1.10422467753342], rtol=1e-9)
    with pytest.raises(ValueError):
        dsp.Filter({"type": "bandpass", "low": 45, "high": 0.5}, RATE)
    with pytest.raises(ValueError):
        dsp.Filter({"type": "magic"}, RATE)


def test_spectra_and_band_power():
    f, p = dsp.periodogram(signal(), RATE)
    assert len(f) == 201 and f[1] == 0.3125
    np.testing.assert_allclose(p[[1, 2]], [0.974906435845878, 0.0474634657599522], rtol=1e-9)
    f, p = dsp.welch(np.stack([signal(), signal()]), RATE, segment=125)
    assert p.shape == (2, 63) and f[1] == 1.0
    np.testing.assert_allclose(p[0, [1, 2]], [0.216959276865522, 0.378819915865643], rtol=1e-9)
    t = np.arange(1000) / RATE
    f, p = dsp.welch(3 * np.sin(2 * np.pi * 10 * t), RATE, segment=250)
    assert abs(dsp.band_power(f, p, 8, 12) - 4.5) < 0.05 and dsp.band_power(f, p, 20, 30) < 1e-6


def test_envelope_wavelet_coherence_and_coupling():
    amplitude, phase = dsp.analytic(signal())
    np.testing.assert_allclose(amplitude[[0, 150]], [7.05107240995648, 2.62587369426418], rtol=1e-9)
    np.testing.assert_allclose(phase[[0, 150]], [1.51109114458893, -0.0665963858101306], rtol=1e-9)
    t = np.arange(1000) / RATE
    slow, fast = np.sin(2 * np.pi * 6 * t), 0.3 * np.sin(2 * np.pi * 40 * t)
    amplitude, _ = dsp.morlet(3 * np.sin(2 * np.pi * 10 * t) + 500, RATE, 10)
    assert abs(amplitude[500] - 3) < 0.01
    theta = dsp.Filter({"type": "bandpass", "low": 4, "high": 8}, RATE)
    gamma = dsp.Filter({"type": "bandpass", "low": 30, "high": 50}, RATE)

    def measure(x):
        return dsp.coupling(dsp.analytic(theta.filtfilt(x))[1][125:-125], dsp.analytic(gamma.filtfilt(x))[0][125:-125])

    coupled, uncoupled = measure(slow + (1 + 0.8 * slow) * fast), measure(slow + fast)
    assert abs(coupled.modulation_index - 0.0481399776253) < 1e-7 and abs(coupled.mean_vector_length - 0.322302472461) < 1e-7
    assert uncoupled.modulation_index < 1e-8 and uncoupled.mean_vector_length < 1e-4
    k = np.arange(400)
    other = np.sin(0.1 * k + 0.5) + 0.3 * np.sin(0.7 * k) + 0.2 * np.sin(0.013 * k * k)
    f, c = dsp.coherence(signal(), other, RATE, segment=100)
    assert len(f) == 51 and f[1] == 1.25
    np.testing.assert_allclose(c[[1, 2, 10]], [0.912333094227104, 0.992570765692503, 0.0139922548019482], rtol=1e-8)


def test_heartbeats():
    beats = [0.5]
    while beats[-1] < 60:
        beats.append(beats[-1] + 0.8 + 0.05 * np.sin(2 * np.pi * 0.25 * beats[-1]))
    t = np.arange(1500) / 25.0
    ppg = 30000 + 400 * sum(np.exp(-(t - b) ** 2 / (2 * 0.07 ** 2)) for b in beats) + 300 * np.sin(0.1 * t)
    found = dsp.heart(ppg, 25.0)
    assert abs(found.rate_bpm - 75) < 1 and found.quality > 0.97 and len(found.beats) > 70
    assert 20 < found.sdnn_ms < 50
    assert np.isnan(dsp.heart(np.full(500, 3.0), 25.0).rate_bpm)


def test_artifact_reconstruction():
    rng = np.random.default_rng(0)
    mix = np.array([[1.0, 0.6, 0.2, 0.1], [0.5, 1.0, 0.3, 0.2], [0.2, 0.4, 1.0, 0.5], [0.1, 0.2, 0.6, 1.0]])
    reference, data = 20 * mix @ rng.normal(size=(4, 7500)), 20 * mix @ rng.normal(size=(4, 1000))
    mixing, threshold = dsp.asr_calibrate(reference, RATE)
    assert mixing.shape == threshold.shape == (4, 4)
    np.testing.assert_allclose(mixing @ mixing, np.cov(reference, bias=True), rtol=0.15, atol=30)
    dirty = data.copy()
    dirty[1, 400:525] += 600 * np.sin(2 * np.pi * 3 * np.arange(125) / RATE)
    asr = dsp.Asr(mixing, threshold)
    cleaned = asr(dirty, RATE)
    error = lambda x: np.sqrt(((x - data)[1, 400:525] ** 2).mean())
    assert error(cleaned) < 0.3 * error(dirty)
    np.testing.assert_allclose(cleaned[:, :300], dirty[:, :300], atol=1e-9)
    asr.reset()
    np.testing.assert_allclose(asr(dirty, RATE), cleaned, atol=1e-9)
    with pytest.raises(ValueError):
        dsp.asr_calibrate(reference[:, :100], RATE)
    with pytest.raises(ValueError):
        asr(dirty[:3], RATE)


def test_pipeline_steps_match_the_reference():
    model = openfrenz.Pipeline(FIXTURE)
    assert (model.name, model.outputs, model.inputs) == ("fixture", ("a", "b"), {"eeg": 250, "accel": 100})
    assert model.folder == FIXTURE
    expected = json.loads((FIXTURE / "expected.json").read_text())["outputs"]
    for run in range(2):  # the same again after a reset
        for index, want in enumerate(expected):
            got = model.step(fixture_step(index))
            if want is None:
                assert got is None
            else:
                np.testing.assert_allclose([got["a"], got["b"]], want, rtol=1e-5)
        model.reset()
    with pytest.raises(ValueError):
        model.step({"eeg": np.zeros((4, 10)), "accel": np.zeros((3, 100))})
    with pytest.raises(ValueError):
        openfrenz.Pipeline(FIXTURE.parent)  # a folder without a config.json


def test_pipeline_runs_on_a_recording_and_on_a_band(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "openfrenz_model": "pipeline", "name": "alpha", "version": "1",
        "step": {"stream": "eeg", "samples": 125},
        "nodes": [
            {"name": "clean", "op": "filter", "in": "eeg", "steps": [{"type": "bandpass", "low": 1, "high": 45}]},
            {"name": "power", "op": "band_power", "in": "clean", "bands": [[8, 12]], "relative_to": [1, 45]},
        ],
        "outputs": ["power"],
    }))
    recording = tmp_path / "synthetic.sqlite"
    openfrenz.synth(recording, seconds=12)
    model = openfrenz.Pipeline(tmp_path)
    scores = model.run(recording)
    assert list(scores.columns) == ["t", *model.outputs] and len(model.outputs) == 4
    assert 10 <= len(scores) <= 12 and scores.t.is_monotonic_increasing
    # The synthetic recording has a 10 Hz rhythm on the comb channels and none on the forehead.
    assert scores["comb_left:power:8-12"].median() > 0.5 > scores["forehead_left:power:8-12"].median()
    assert scores.attrs["missed"] == {}
    with openfrenz.replay(recording, speed=4, record=False) as band:
        stream = model.stream(band)
        live = list(stream)
    assert 8 <= len(live) <= 12 and set(live[0]) == {"t", *model.outputs}
    assert np.median([row["comb_left:power:8-12"] for row in live]) > 0.5
    assert stream.missed == {}


def test_a_live_model_says_what_it_is_waiting_for(tmp_path):
    def write(folder, required):
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({
            "openfrenz_model": "pipeline", "step": {"stream": "eeg", "samples": 125},
            "inputs": {"ppg": {"samples": 25, "required": required}},
            "nodes": [{"name": "size", "op": "summary", "in": "eeg", "stats": ["sd"]},
                      {"name": "pulse", "op": "summary", "in": "ppg", "stats": ["mean"]}],
            "outputs": ["size", "pulse"],
        }))
        return openfrenz.Pipeline(folder)

    # A recording of EEG alone: the pulse stream never comes.
    recording = tmp_path / "eeg.sqlite"
    openfrenz.synth(recording, seconds=8)
    import sqlite3

    db = sqlite3.connect(recording)
    counts = dict(db.execute("select char, count(*) from notifications group by char"))
    eeg = max(counts, key=counts.get)  # the EEG characteristic sends the most packets
    db.execute("delete from notifications where char != ?", (eeg,))
    db.commit()
    db.close()

    needs = write(tmp_path / "needs", True)
    with pytest.raises(openfrenz.FrenzError, match="no ppg stream"):
        needs.run(recording)
    with openfrenz.replay(recording, speed=1, record=False) as band:
        stream = needs.stream(band)
        assert stream.read(timeout=5) is None       # silent, and it says why:
        assert stream.missed.get("ppg", 0) >= 1

    prefers = write(tmp_path / "prefers", False)
    scored = prefers.run(recording)
    assert len(scored) >= 6 and scored.attrs["missed"] == {}
    assert scored.filter(like="tag").isna().all().all() and scored.filter(like="forehead").notna().all().all()
