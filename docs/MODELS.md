# Models and signal processing

openfrenz can turn the band's streams into numbers you care about: filter them, measure rhythms,
find heartbeats, remove artifacts, and run a trained model on the result, on a recording or on a
band as it streams. There are three ways in, all running the same Rust code:

| You want to | Use |
|---|---|
| Compute features in Python, for analysis or to train a model | [`openfrenz.dsp`](#blocks-in-python) |
| Run a model that is described entirely by its files | [`openfrenz.Pipeline`](#pipelines-models-as-data) (Python) or `frenz_model::Pipeline` (Rust) |
| Write a model in Rust with any logic you like | the [`Model` trait](RUST.md#run-a-model) in `frenz-model` |

Because training and deployment share one implementation, a feature computed in Python for
training is the same number, to rounding, as the one a deployed pipeline computes live.

!!! note "Units"
    Stream values are raw device counts. The microvolt scale of the EEG is not known, so
    absolute powers are in squared counts; relative powers, coherence and coupling do not depend
    on the scale.

## Blocks in Python

```python
import numpy as np
import openfrenz
from openfrenz import dsp

session = openfrenz.export("recordings/FRENZA12-1795203836.sqlite")
t, eeg, segment = session.signal("eeg")          # eeg is (4, n) raw counts at 125 Hz
```

A signal is a NumPy array: `(n,)` for one channel or `(channels, n)`. Channel order is that of
the stream (`openfrenz.SIGNALS`).

### Filters

```python
clean = dsp.Filter(
    [{"type": "dc_block"},
     {"type": "notch", "hz": 60},
     {"type": "bandpass", "low": 0.5, "high": 45}],
    rate_hz=125, start="settled")

y = clean(eeg)                # filters these samples and remembers where it stopped
y2 = clean(more_eeg)          # continues exactly as if both had been filtered together
clean.reset()                 # start over, e.g. after a hole in the data
smooth = clean.filtfilt(eeg)  # forwards and backwards: no delay, needs the whole signal
```

| Filter | Settings | Notes |
|---|---|---|
| `lowpass` | `high`, `order` (2) | Butterworth; -3 dB at `high` |
| `highpass` | `low`, `order` (2) | Butterworth; -3 dB at `low` |
| `bandpass` | `low`, `high`, `order` (2) | Butterworth; has `2 * order` poles, as in SciPy |
| `notch` | `hz`, `q` (30) | `q` is the centre frequency over the notch's width |
| `dc_block` | `pole` (0.99) | removes a constant offset; a pole nearer 1 settles more slowly |

Filters keep their state, which is what a live model needs: filtering a signal 2 seconds at a
time gives exactly the numbers that filtering it whole gives. `start="settled"` begins as if the
first sample had always been there, which avoids a large transient when the signal has an offset
(raw EEG counts do); `start="zero"` (the default) begins from silence, as SciPy does. The numbers
match SciPy's `butter`, `iirnotch`, `sosfilt` and `sosfiltfilt`.

### Spectra and band power

```python
freqs, density = dsp.welch(y, 125, segment=250)     # averaged over 2-second stretches
alpha = dsp.band_power(freqs, density, 8, 12)        # one value per channel
relative = alpha / dsp.band_power(freqs, density, 1, 45)
freqs, density = dsp.periodogram(y[:, :250], 125)    # a single stretch: noisier
```

Density is power per hertz, one-sided, with the mean removed and a Hann window, as SciPy's
`periodogram` and `welch`. `band_power` sums the bins from the lower edge (included) to the upper
(excluded), so adjacent bands do not share a bin.

### Wavelets

```python
amplitude, phase = dsp.morlet(y[0], 125, hz=10, cycles=6)
```

`morlet` follows a rhythm in time: the amplitude at `hz` for every sample, in signal units (a sine
of amplitude 3 reads 3). More cycles separate neighbouring frequencies better and blur time more;
5 to 7 is usual. With 6 cycles a wavelet still sees about 1% of a rhythm an octave away. The first
and last `cycles / hz` seconds read low, because the wavelet runs off the end of the signal.

### Envelope and phase

```python
beta = dsp.Filter({"type": "bandpass", "low": 20, "high": 30}, 125)
amplitude, phase = dsp.analytic(beta.filtfilt(y[0]))
```

`analytic` gives the amplitude envelope and the phase of a signal confined to a narrow band (as
SciPy's `hilbert`), so filter first. Its first and last few cycles are distorted.

### Coherence between channels

```python
freqs, coh = dsp.coherence(y[2], y[0], 125, segment=125)   # forehead_left, forehead_right
```

Coherence is 0 for unrelated signals and 1 for a fixed linear relation, per frequency. It is
computed from several stretches of signal: a single stretch always reads 1, and a handful read
too high. Use at least 8 to 10 stretches (here, 5 seconds or more for 1-second stretches).

### Coupling between rhythms

```python
theta = dsp.Filter({"type": "bandpass", "low": 4, "high": 8}, 125)
gamma = dsp.Filter({"type": "bandpass", "low": 30, "high": 45}, 125)
_, phase = dsp.analytic(theta.filtfilt(x))
amplitude, _ = dsp.analytic(gamma.filtfilt(x))
c = dsp.coupling(phase[125:-125], amplitude[125:-125])      # leave out the distorted ends
c.modulation_index, c.mean_vector_length, c.preferred_phase
```

Phase-amplitude coupling asks whether a fast rhythm is larger at a particular phase of a slow
one. Two measures are returned: Tort's modulation index (0 for none; a few hundredths is already
strong) and a mean vector length scaled to lie between 0 and 1 whatever the signal's size. Both
need many cycles of the slow rhythm, so several seconds of signal, and both respond to sharp
artifacts such as blinks and muscle bursts, which have slow and fast parts locked together.
Clean the signal first and treat a high value on a noisy stretch with suspicion.

### Heartbeats from the pulse sensor

```python
t, ppg, _ = session.signal("ppg")                 # (2, n) at 25 Hz
h = dsp.heart(ppg[0], 25)
h.rate_bpm, h.sdnn_ms, h.rmssd_ms, h.quality, h.beats
```

`heart` band-passes the pulse signal to 0.7–3.5 Hz, finds each beat as a rise, and times it
between samples. It expects pulses that point up; pass `-ppg` if yours point down.
The two pulse channels can differ a lot in how clean they are, so try each and keep the one with the
higher `quality`.

| Value | Meaning |
|---|---|
| `beats` | time of each beat, seconds from the start of the signal |
| `rate_bpm` | beats per minute, from the mean interval |
| `sdnn_ms` | standard deviation of the intervals between beats |
| `rmssd_ms` | root mean square of the change from one interval to the next |
| `quality` | share of intervals that look like heartbeats (plausible length, close to the median, between beats of ordinary size) |

Distrust everything else when `quality` is below about 0.8: movement easily swamps the pulse.
Variability needs a long stretch (a minute or more for SDNN), and the band's 25 Hz pulse signal
times each beat to a few milliseconds at best, which adds that much to both variability measures.
They are fine for comparing conditions and not a substitute for an ECG.

### Removing artifacts

Artifact subspace reconstruction removes large, brief disturbances (blinks, jaw clenches, cable
tugs) from multi-channel data. It learns from a quiet stretch how large each direction of the
signal normally is; later, directions that are far larger than that are removed and rebuilt from
the others.

```python
clean = dsp.Filter([{"type": "dc_block"}, {"type": "bandpass", "low": 0.5, "high": 45}], 125, start="settled")
quiet = clean(eeg[:, :125 * 60])                       # a minute of quiet data, filtered
mixing, threshold = dsp.asr_calibrate(quiet, 125, cutoff=20)

asr = dsp.Asr(mixing, threshold)
cleaned = asr(clean(eeg[:, 125 * 60:]), 125)           # keeps its state: call it step after step
```

- **Calibrate on data filtered the same way** as the data you will clean, and without drift or
  offset.
- **`cutoff`** is how many times its usual spread a direction may exceed its usual size before it
  is removed. 20 to 30 is the recommended range (Chang et al. 2020); lower values remove more and
  start to remove brain signal.
- **A calibration belongs to its recording conditions.** Different people, fits and days give
  different signals; a calibration made on one is not right for another.
- **Four channels limit what can be rebuilt.** With few channels, a removed direction is rebuilt
  from little, so a disturbance is reduced, not erased: in our test a slow wave 25 times the
  signal's size on one channel is reduced about fivefold.

This is a compact form of the published method (Mullen et al. 2015; Kothe & Jung). It differs from
the EEGLAB implementation in three ways: limits come from the median and spread of half-second
sizes, not a fitted distribution; each half-second window is judged by its own data with no
look-ahead, so there is no delay; and there is no built-in spectral weighting, so filter first.

## Pipelines: models as data

A pipeline is a model written entirely in two files: `config.json`, which lists blocks and how
they connect, and `model.safetensors`, which holds any stored numbers (normalisation, artifact
calibration, network weights). It needs no code of its own, so any openfrenz installation can run
it.

```json
{
  "openfrenz_model": "pipeline",
  "name": "example", "version": "1",
  "step": { "stream": "eeg", "samples": 250 },
  "inputs": { "accel": { "samples": 100 } },
  "nodes": [
    { "name": "clean", "op": "filter", "in": "eeg", "start": "settled",
      "steps": [{ "type": "notch", "hz": 60 }, { "type": "bandpass", "low": 0.5, "high": 45 }] },
    { "name": "power", "op": "band_power", "in": "clean",
      "bands": [[4, 8], [8, 12], [12, 30]], "relative_to": [1, 45] },
    { "name": "coh", "op": "coherence", "in": "clean",
      "pairs": [["forehead_left", "forehead_right"]], "bands": [[8, 12]], "average_steps": 5 },
    { "name": "move", "op": "summary", "in": "accel", "stats": ["sd"] },
    { "name": "x", "op": "concat", "in": ["power", "coh", "move"] },
    { "name": "z", "op": "normalize", "in": "x", "weights": "norm" },
    { "name": "recent", "op": "history", "in": "z", "steps": 5 },
    { "name": "net", "op": "network", "in": "recent", "names": ["score"],
      "layers": [{ "type": "lstm", "weights": "rnn" }, { "type": "last" },
                 { "type": "dense", "weights": "head" }, { "type": "sigmoid" }] }
  ],
  "outputs": ["net"]
}
```

How to read it:

- **`step`** names the stream that sets the pace and how many of its samples make one step (here
  2 seconds of EEG). **`inputs`** lists the other streams and how many samples of each belong to a
  step; see [Streams a model can do without](#streams-a-model-can-do-without) for the other
  forms.
- **`nodes`** run in order. Each has a `name`, the block it is (`op`), what it reads (`in`: a
  stream or an earlier node, or a list of them) and the block's settings.
- **`outputs`** names the nodes whose values the model returns.
- **Stored numbers** are named by a node's `weights`, a prefix into `model.safetensors`.

Three kinds of thing flow between nodes: a *signal* (channels by samples), *features* (named
numbers describing one step) and a *sequence* (the features of several steps).

### Run a pipeline

=== "Python"

    ```python
    import openfrenz

    model = openfrenz.Pipeline("my-model")          # a folder, or "owner/name" on Hugging Face
    model.inputs                                    # {'eeg': 250, 'accel': 100}
    model.outputs                                   # ('score',)

    scores = model.run("recordings/session.sqlite") # DataFrame: t and one column per output

    with openfrenz.connect("FRENZA12") as band:
        for out in model.stream(band):              # {'t': ..., 'score': ...} every step
            print(out)

    out = model.step({"eeg": eeg_block, "accel": accel_block})   # one step by hand; None while warming up
    ```

=== "Rust"

    ```rust
    use frenz_model::{Live, Pipeline, run_recording};

    let model = Pipeline::from_dir(Path::new("my-model"))?;
    let run = run_recording(model, Path::new("session.sqlite"))?;       // run.outputs: Vec<Output { t, values }>

    let mut live = Live::new(Pipeline::from_dir(Path::new("my-model"))?, &band, true)?;
    while let Some(output) = live.next().await? { /* ... */ }
    ```

`run` scores a recording as fast as the model computes. `stream` gives one result per step as
the band delivers data. Both restart the model wherever data is missing (a new segment, a hole
inside a step), so nothing is carried across a gap, and both leave out repeated and frozen
packets. Live, gaps are recognised by when packets arrive: replay a recording at its recorded
pace to see what a live run would have given, and use `run` to score it quickly.

A step is skipped when a stream the model requires is missing from it. Skipped steps are counted
by the stream that was missing, so a model that stays silent can say why:

```python
stream = model.stream(band)
out = stream.read(timeout=10)        # None if no step completed in 10 seconds
stream.missed                        # {'ppg': 4}: four steps skipped for want of the pulse stream
scores.attrs["missed"]               # the same count for a scored recording
```

In Rust the counts are `Live::missed()` and the `missed` field of what `run_recording` returns.
The pulse sensor, for example, sends nothing until the band is on the head.

### Streams a model can do without

By default every stream in `inputs` is required, and a step without one is skipped and restarts
the model. Two other forms avoid that:

```json
"inputs": {
  "accel":   { "samples": 100 },
  "ppg":     { "samples": 50, "required": false },
  "battery": { "arrived": true }
}
```

- **`"required": false`** uses the stream when it is there. While it is missing, every feature
  made from it is not-a-number and the rest of the model carries on, without a restart. Blocks
  that gather several steps of it start gathering again when it returns, and give not-a-number
  until they have enough. Use this for streams that drop out on their own, such as the pulse
  sensor, which freezes for a second or two now and then.
- **`"arrived": true`** is for streams without a steady rate (`heart_rate`, `spo2`, `battery`,
  `contact`): a step gets whatever readings arrived during it, often none. `summary` of nothing is
  not-a-number.

Not-a-number travels through `concat`, `normalize`, `history` and `network`, so keep optional
features out of a network's inputs unless the network was trained to be given a stand-in value.

A pipeline is checked when it is loaded, by running it on silence. A block given the wrong kind
or amount of input is reported then, with the node's name:

```
config.json: node z: was stored for 35 features and given 18
```

### Blocks

| Block | Reads → gives | Settings (defaults) |
|---|---|---|
| `select` | signal → signal | `channels`: names to keep, in order |
| `filter` | signal → signal | `steps`: filters as in [Filters](#filters); `start`: `zero` or `settled` (`zero`); `zero_phase`: filter each step both ways, without state (`false`) |
| `asr` | signal → signal | `weights`: prefix of `mixing` and `threshold`; `max_removed` (0.66); `window_s` (0.5) |
| `band_power` | signal → features | `bands`: list of `[low, high]`; `relative_to`: `[low, high]` to divide by; `log` (`false`); `segment`: average over stretches of that many samples |
| `wavelet` | signal → features | `frequencies`: list of Hz; `cycles` (6); `trim_s`: seconds left out at each end (0); `log` (`false`). Gives the mean amplitude over the step |
| `envelope` | signal → features | `band`: `[low, high]`; `order` (2); `stats` (`["mean", "sd"]`) |
| `summary` | signal → features | `stats`: any of `mean`, `sd`, `range`, `largest_step` (`["mean", "sd"]`) |
| `quality` | signal → features | `limit`: the largest value the sensor reports. Gives the shares of repeated, clipped and invalid samples |
| `coherence` | signal → features | `pairs`: list of `[channel, channel]`; `bands`; `segment` (125); `average_steps` (1) |
| `coupling` | signal → features | `phase_band`, `amplitude_band`; `order` (2); `bins` (18); `trim_s` (0); `average_steps` (1) |
| `heart` | signal → features | `channel` (the first); `invert` (`false`); `window_steps` (1). Gives rate, SDNN, RMSSD and quality |
| `concat` | features, ... → features | |
| `normalize` | features → features | `weights`: prefix of stored `center` and `scale`; or `half_life_steps` and `warmup_steps` (2) to track them as steps pass |
| `history` | features → sequence | `steps`: how many to keep |
| `network` | features or sequence → features | `layers`; `names`: one per output value |

Things to know:

- **State.** `filter`, `asr`, `coherence`, `coupling`, `heart`, `history` and a running
  `normalize` carry state from step to step, and all of it is cleared at a reset.
- **Warm-up.** A block that needs several steps (`average_steps`, `window_steps`, `steps`,
  `warmup_steps`) gives nothing until it has them, and neither does anything that reads it. The
  model's first output comes when every block is ready.
- **Averaging over steps.** Coherence and coupling cannot be measured in one short step. With
  `average_steps: 5` and 2-second steps they describe the last 10 seconds, updated every step.
- **Feature names** are built from the channel and the measure, for example
  `forehead_left:power:8-12`, `forehead_left~forehead_right:coherence:8-12`, `x:sd`. A model's
  `outputs` lists them.
- **Network layers** are `dense` and `lstm` (each with `weights`, and `dense` optionally with an
  `activation`), `last`, `mean` and `flatten` to turn a sequence into one row, and `relu`, `elu`,
  `tanh`, `sigmoid` and `gelu`. Arrays are laid out as PyTorch stores them: `prefix.weight` and
  `prefix.bias` for `dense`; `prefix.weight_ih_l0`, `prefix.weight_hh_l0`, `prefix.bias_ih_l0` and
  `prefix.bias_hh_l0` for `lstm`. The network must end in a single row.

### Write a pipeline from Python

Train with features from `openfrenz.dsp`, then save the same steps as a pipeline. With PyTorch and
the `safetensors` package:

```python
import json
import torch
from safetensors.torch import save_file

# rnn = torch.nn.LSTM(features, 16, batch_first=True); head = torch.nn.Linear(16, 1)  ... trained
weights = {f"rnn.{k}": v for k, v in rnn.state_dict().items()}
weights |= {f"head.{k}": v for k, v in head.state_dict().items()}
weights |= {"norm.center": torch.tensor(train_mean), "norm.scale": torch.tensor(train_sd)}
save_file(weights, "my-model/model.safetensors")

with open("my-model/config.json", "w") as f:
    json.dump(config, f, indent=1)                  # the dict shown above

openfrenz.Pipeline("my-model")                      # loads and checks it
```

An artifact calibration is stored the same way, as `prefix.mixing` and `prefix.threshold` from
`dsp.asr_calibrate`. Compare the pipeline's outputs with your training code on a few steps before
relying on it: `model.step(...)` makes that a few lines.

### Share a model on Hugging Face

Upload the folder (`config.json`, `model.safetensors`, a README) to a model repository, and load
it by name:

```python
model = openfrenz.Pipeline("owner/name")            # or "owner/name@v1.0", "owner/name@<commit>"
```

Files are downloaded once per commit into `~/.cache/openfrenz` (set `OPENFRENZ_CACHE` to move
it); a branch name is looked up each time to see which commit it points to. Private repositories
use the token in `HF_TOKEN`, or the one saved by the Hugging Face command-line login. The files hold numbers and
settings only and are never executed, and the sizes a `config.json` can ask for are bounded
(samples per step, filter order, steps kept, and so on), so a model from an unknown source can
be slow or wrong but cannot run code. Model files over 512 MB are not downloaded.

## Models in code

A pipeline covers models built from the standard blocks. For anything else, write the model in
Rust against the `Model` trait: it declares the streams it reads and is called once per step. Such
a model can read any stream the band sends, including the slow ones (`heart_rate`, `spo2`,
`battery`, `contact`), can mark a stream as preferred, and can decide for itself what to do when
one is absent.
See [Run a model](RUST.md#run-a-model) in the Rust guide.

## Limits

- Pipelines run on `.sqlite` recordings and on bands, not yet on exported session folders.
- The blocks are tested against SciPy, NumPy and PyTorch on synthetic signals. Artifact
  reconstruction and heartbeat detection have not been validated against reference recordings
  (simultaneous research-grade EEG or ECG).
- Running a pipeline live has been checked on replayed recordings and, briefly, on one band.
