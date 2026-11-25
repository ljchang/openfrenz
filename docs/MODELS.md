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
| `mix` | signal → signal | `channels`: list of `{"name": ..., "weights": {channel: weight}}`. Each new channel is a weighted sum, for example a difference between two electrodes |
| `filter` | signal → signal | `steps`: filters as in [Filters](#filters); `start`: `zero`, `settled`, or a number (that value, held for ever) (`zero`); `zero_phase`: filter each step both ways, without state (`false`) |
| `asr` | signal → signal | `weights`: prefix of `mixing` and `threshold`; `max_removed` (0.66); `window_s` (0.5); `weighting`: `{"b": [...], "a": [...]}`; `average_windows` (1); `blend_offset`: 1 or 0 (1); `before`: filters run inside the node; `before_start`; `passes` (1). See [Matching another cleaner](#matching-another-cleaner) |
| `band_power` | signal → features | `bands`: list of `[low, high]`; `relative_to`: `[low, high]` to divide by; `log` (`false`); `segment`: average over stretches of that many samples; `measure`: `power` or `magnitude` (`power`); `integrate`: `sum` or `simpson` (`sum`); `window`: `hann` or `none` (`hann`); `remove_mean` (`true`). See [Shares, ratios and averages](#shares-ratios-and-averages) |
| `wavelet` | signal → features | `frequencies`: list of Hz; `cycles` (6); `trim_s`: seconds left out at each end (0); `log` (`false`). Gives the mean amplitude over the step |
| `envelope` | signal → features | `band`: `[low, high]`; `order` (2); `stats` (`["mean", "sd"]`) |
| `summary` | signal → features | `stats`: any of `mean`, `sd`, `log_sd`, `range`, `largest_step` (`["mean", "sd"]`) |
| `quality` | signal → features | `limit`: the largest value the sensor reports. Gives the shares of repeated, clipped and invalid samples |
| `coherence` | signal → features | `pairs`: list of `[channel, channel]`; `bands`; `segment` (125); `average_steps` (1) |
| `phase_amplitude` | signal → features | `phase_band`, `amplitude_band`; `order` (2); `trim_s` (0); `stats`: any of `amplitude_mean`, `amplitude_sd`, `phase_mean`, `phase_sd`, `correlation` (all) |
| `coupling` | signal → features | `phase_band`, `amplitude_band`; `order` (2); `bins` (18); `trim_s` (0); `average_steps` (1) |
| `heart` | signal → features | `channel` (the first); `invert` (`false`); `window_steps` (1). Gives rate, SDNN, RMSSD and quality |
| `peaks` | signal → features | `threshold` (4); `back_s` (2); `ahead_s` (0.25); `min_gap_s` (0.25); `scale_half_life_s` (120); `warmup_s` (5). Gives `peaks`, `peak_time` and `peak_size`; see [Moments](#moments-peaks-and-jumps) |
| `jumps` | signal → features | `threshold` (6); `leak_s` (5); `min_gap_s` (0.12); `near_s` (0.048); `far_s` (0.152); `scale_half_life_s` (120); `warmup_s` (5). Gives `position` and `jumps` |
| `tilt` | signal → features | `counts_per_g` (16384). Reads the accelerometer's `x`, `y`, `z`; gives `pitch_deg`, `roll_deg`, `from_upright_deg` and `g`; see [Head posture and movement](#head-posture-and-movement) |
| `ratios` | features → features | `pairs`: list of `[measure, measure]`; `log` (`false`) |
| `average` | features → features | `channels`: which to average (all); `name` (`mean`) |
| `concat` | features, ... → features | |
| `normalize` | features → features | `weights`: prefix of stored `center` and `scale`; or `half_life_steps` and `warmup_steps` (2) to track them as steps pass |
| `history` | features → sequence | `steps`: how many to keep |
| `network` | features or sequence → features | `layers`; `names`: one per output value |

Things to know:

- **State.** `filter`, `asr`, `coherence`, `coupling`, `heart`, `peaks`, `jumps`, `history` and a
  running `normalize` carry state from step to step, and all of it is cleared at a reset.
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

### Moments: peaks and jumps

Two blocks find things that happen at a moment instead of describing a whole step. Both work
sample by sample, so the result does not depend on how long the steps are.

**`peaks`** finds peaks that stand out from their surroundings: local maxima whose height above
the higher of the two lowest points nearby (within `back_s` before and `ahead_s` after) is at
least `threshold` times the signal's ordinary size. The ordinary size is a running median of the
signal's absolute value, so the peaks themselves do not raise it; it follows slow changes over
about `scale_half_life_s`. Of peaks closer together than `min_gap_s`, only the highest counts.
Each step gives, per channel:

| Feature | Meaning |
|---|---|
| `peaks` | how many peaks became certain during this step |
| `peak_time` | when the largest of them was, in seconds counted back from the step's last sample (zero or negative; not-a-number if none) |
| `peak_size` | how far it stood out, in units of the ordinary size (not-a-number if none) |

A peak is reported `ahead_s` to `ahead_s + min_gap_s` after it happened, so it often belongs to
an earlier step than the one that reports it. Its time is the step's `t` plus `peak_time`:

```python
out = model.run("recordings/session.sqlite")
found = out[out["blink:peaks"] > 0]
blink_times = found["t"] + found["blink:peak_time"]
```

**`jumps`** follows a level that moves in abrupt steps while the sensor's baseline wanders. A
change between two samples counts as a jump when it is `threshold` times the ordinary change; its
height is the mean level from `near_s` to `far_s` after it minus the same before it. Heights are
added to a `position` that fades back to zero, losing half of itself every `leak_s`. Slow wander
never reaches the position, and a level that changes and stays changed is still seen, which a
high-pass filter would lose. `jumps` counts the jumps added during the step, each `far_s` after
it happened.

Neither block detects anything for the first `warmup_s`, while it learns the ordinary size.

### Matching another cleaner

The `asr` block's options let a pipeline reproduce artifact subspace reconstruction as other software runs
it, sample for sample:

- **`weighting`** filters the data before its size is judged (the cleaned data itself is not filtered), as
  EEGLAB does with its spectral weighting. Give the filter's coefficients, `b` over `a`; it starts settled on
  the first sample.
- **`average_windows`** judges each window by the average covariance of it and the windows before it.
- **`blend_offset`** sets where the ease from one window's reconstruction into the next starts: 1 (EEGLAB) or 0.
- **`before`** runs a filter chain inside the node, before the reconstruction, starting as `before_start`
  says. On its own it is the same as a `filter` node in front.
- **`passes`** runs that chain and the reconstruction several times over each step and keeps the first result,
  so both see each step that many times. Some pipelines were trained this way, and their models need it to
  give the same numbers.
- **`max_removed`** of 0.75 with four channels always keeps one direction.

### Shares, ratios and averages

Three blocks turn per-channel spectra into the kind of numbers a model is usually given:

- **`band_power`** gives power per band by default. With `"measure": "magnitude"` it gives the size of the
  Fourier transform instead of its square, over the whole step without averaging. `"integrate": "simpson"`
  integrates each band with Simpson's rule over the bins nearest its edges, both included, instead of summing
  bins. With `relative_to`, each band is a share of that range.
- **`ratios`** divides one measure by another for every channel that has both. The pair
  `["magnitude:4-8", "magnitude:8-12"]` turns `forehead_left:magnitude:4-8` and
  `forehead_left:magnitude:8-12` into `forehead_left:magnitude:4-8/magnitude:8-12`, and the same for each other
  channel. Any two measures can be paired, from any blocks, as long as both reach the `ratios` node.
- **`average`** averages each measure over channels: `forehead_left:x` and `forehead_right:x` become
  `mean:x`. Averaging after `ratios` gives the mean of the ratios; averaging the signals first would not.

```json
{ "name": "band", "op": "filter", "in": "front", "zero_phase": true,
  "steps": [{ "type": "bandpass", "low": 2, "high": 45, "order": 1 }] },
{ "name": "shares", "op": "band_power", "in": "band", "bands": [[4, 8], [8, 12]], "relative_to": [0, 62],
  "measure": "magnitude", "integrate": "simpson", "window": "none" },
{ "name": "ratio", "op": "ratios", "in": "shares", "pairs": [["magnitude:4-8", "magnitude:8-12"]] },
{ "name": "mean", "op": "average", "in": ["ratio"] }
```

**`phase_amplitude`** relates the phase of one band to the amplitude envelope of another within each step: the
envelope's mean and spread, the phase's mean and spread after removing its jumps of 2π, and the correlation
between the two. Both bands are filtered both ways, so the step's ends are slightly distorted; `trim_s`
leaves them out. `coupling` measures something different: how the envelope's size depends on the phase,
over several steps.

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
it); a branch name is looked up each time to see which commit it points to, and when Hugging Face
cannot be reached the commit it pointed to last time is used, so a model fetched once also works
offline.

Private repositories need a token from an account with access. Give it as `token`, or let
openfrenz find one in `HF_TOKEN` or the Hugging Face command-line login:

```python
model = openfrenz.Pipeline("lab/private-model", token=my_token)
```

A repository that cannot be read (private without a token with access, or not there) raises an
error that says which; `HF_ENDPOINT` points openfrenz at a mirror. The files hold numbers and
settings only and are never executed, and the sizes a `config.json` can ask for are bounded
(samples per step, filter order, steps kept, and so on), so a model from an unknown source can
be slow or wrong but cannot run code. Model files over 512 MB are not downloaded.

## Models that come with openfrenz

Five pipelines are built in and are loaded by name. None needs stored numbers.
`openfrenz.MODELS` lists them.

| Name | Reads | Gives, per step |
|---|---|---|
| `eog` | `eeg` | eye signals: blinks and horizontal gaze, every 0.2 s |
| `brow-emg` | `eeg` | muscle activity of the forehead, every 0.2 s |
| `heart` | `ppg` | heart rate, its variability and a quality score over the last 30 s, every 5 s |
| `head` | `accel`, `gyro` | how the head is held and how much it moves and turns, every second |
| `eeg-power` | `eeg` | power in the delta, theta, alpha, beta and gamma bands of each electrode and their average, every second |

**Live**, as the band streams:

```python
model = openfrenz.Pipeline("eog")
with openfrenz.connect("FRENZA12") as band:
    for out in model.stream(band):
        if out["blink:peaks"]:
            print("blink at", out["t"] + out["blink:peak_time"])
```

**In an exported session**, as one more table beside the streams:

```python
session = openfrenz.export("recordings/session.sqlite", models=["eog", "brow-emg", "heart"])
session["eog"]                              # DataFrame: t and one column per output
```

```sh
frenz export recordings/session.sqlite --model eog --model brow-emg --model heart
```

`models` and `--model` also take a folder or a Hugging Face repository, so your own pipelines
are exported the same way. The recording itself is never changed: it stays the raw record, and a
session can be exported again whenever a model improves.

### Which model made a table

These models will change as more data is collected. Each model table in `session.json` records
where its numbers came from:

```python
session.meta["tables"]["eog"]["model"]
# {'name': 'eog', 'version': '0.1', 'source': 'eog',
#  'fingerprint': '<16 hex digits>', 'runner': 'openfrenz frenz-model 0.3.0',
#  'outputs': [...], 'missed': {'eeg': 1}}
```

- `version` is what the model's `config.json` calls itself.
- `fingerprint` is computed from the model's files and changes whenever they do, even if nobody
  remembered to change the version.
- `runner` is the openfrenz version that ran it; `session.meta["exporter"]` is the one that
  exported the streams.

To find sessions made with an older model, compare with the model in hand:

```python
current = openfrenz.Pipeline("eog").fingerprint
stale = [p for p in openfrenz.find_sessions("sessions")
         if openfrenz.load_session(p).meta["tables"].get("eog", {}).get("model", {}).get("fingerprint") != current]
```

### Eye and brow signals

The eyes and the forehead muscles are read from the same electrodes in different frequency
ranges. They are two models, so that each can change on its own. Both give a row every 0.2 s.

| Model | Output | From | Meaning |
|---|---|---|---|
| `eog` | `blink:peaks`, `blink:peak_time`, `blink:peak_size` | forehead pads minus side combs, 0.5-8 Hz, `peaks` | blinks and longer eye closures |
| `eog` | `gaze_horizontal:position`, `gaze_horizontal:jumps` | left minus right forehead pad, below 8 Hz, `jumps` | where the eyes point, left to right, in device counts; fades to zero within seconds |
| `brow-emg` | `brow:log_sd` | left minus right forehead pad, 20-58 Hz, `summary` | how hard the forehead muscles are working (natural logarithm) |

**These outputs are experimental.** They were checked against face video analysed with
[py-feat](https://py-feat.org), on two ten-minute sessions of one person on consecutive days:

| Output | Check | Day 1 | Day 2 |
|---|---|---|---|
| Blinks | F1 against blinks seen on video, within 0.2 s | 0.49 | 0.70 |
| Horizontal gaze | correlation with gaze on video, during cued looks left and right | 0.89 | 0.69 |
| Brow | correlation with brow raise on video, whole session | 0.59 | 0.56 |

Blink times were within 15 ms of the video on both days. Version 0.2 of `eog` lowered the blink threshold
from 4 to 3: on a third day, when the side combs were about 45 times noisier than the forehead pads, version
0.1 found 9 of 15 cued natural blinks and 0.2 finds all 15. The cost is more false detections on quieter days
(F1 0.50 to 0.49 and 0.76 to 0.70 on the two video days). What to keep in mind:

- **Gaze is readable only while the face is still.** Over a whole session, including posed
  expressions, its correlation with video is near zero: one large movement of the forehead adds
  a jump far bigger than any eye movement. It gives direction and relative size, not degrees.
- **Vertical gaze is not given.** The band's electrodes do not separate it.
- **Brow level is relative.** Compare it with a stretch of rest from the same session, or follow
  it with a running `normalize`.
- **Blink detection depends on the side combs.** The blink signal is the forehead pads minus the combs, and how
  noisy the combs are varies with fit from day to day.
- One person and two days cannot show how these hold across people, hair or fit.

### EEG power bands

`eeg-power` filters each electrode (60 Hz notch, 1-45 Hz band-pass) and gives, every second, the natural
logarithm of its power in 1-4 (delta), 4-8 (theta), 8-13 (alpha), 13-30 (beta) and 30-45 Hz (gamma), named
like `forehead_left:power:8-13`, plus each band averaged over the four electrodes (`mean:power:8-13`).

On this band most of what is above about 20 Hz is muscle, not brain (see the eye and brow findings), so
read beta and especially gamma as facial tension unless the face is still. Power is in squared device counts:
compare it within a session, not between bands or people.

### Heart rate and its variability

`heart` runs the [`heart` block](#heartbeats-from-the-pulse-sensor) on one pulse channel (`tag3`)
over the last 30 s and gives `heart:rate_bpm`, `heart:sdnn_ms`, `heart:rmssd_ms` and
`heart:quality` every 5 s.

**Also experimental, and not checked against an ECG.** On the same two sessions `quality` was
above 0.8 for 96% and 69% of steps, with a median rate of 72 and 71 beats a minute; the band's
own heart-rate stream, which reported a value in only 2% and 10% of its readings, averaged 78
and 73. Keep only steps with `quality` above about 0.8. The other pulse channel (`tag2`) was
unusable in both sessions. Thirty seconds is short for variability, and 25 Hz limits how finely
each beat is timed; see the block's notes.

### Head posture and movement

`head` reads the motion sensors and gives, every second:

| Output | Meaning |
|---|---|
| `tilt:pitch_deg` | how far the head leans forward (positive, chin down) or back, from the direction of gravity |
| `tilt:roll_deg` | how far it leans toward the wearer's left shoulder (positive) or right |
| `tilt:from_upright_deg` | angle between the top of the head and straight up: near 0 upright, near 90 lying down |
| `tilt:g` | size of the accelerometer reading in g: 1 when still |
| `accel_down:sd`, `accel_left:sd`, `accel_forward:sd` | how much the head was jolted along each axis during the second (above 0.5 Hz, in counts) |
| `turn_shake:sd`, `turn_nod:sd`, `turn_side:sd` | how much it turned about the vertical axis (a shake), the ear-to-ear axis (a nod) and the nose axis (ear toward shoulder), in gyroscope counts |

What these can and cannot say:

- **Upright or lying down, and on which side.** Lying on the back gives a pitch near -90, face
  down near +90, on the right side a roll near -90, on the left near +90.
- **Sitting and standing look the same**: the head is upright in both. Telling them apart needs
  more than how the head is held, for example the moment of standing up.
- **The band sits at an angle on the forehead**, and not the same angle every time: sitting
  upright read 12 and 23 degrees from upright on two days. Compare with a known upright stretch
  from the same session before reading small leans.
- **Nods and shakes are not counted yet.** The turn outputs say how much the head turned about
  each axis, which is what published detectors of nods and shakes build on, but no recording
  with cued nods and shakes has been made to set and check a detector. The sensor's y and z axes
  are also rotated about 30 degrees from the band's, so nods and sideways tilts are not cleanly
  separated; shakes, about the vertical axis, are.

The angles follow from the sensor's axes and gravity and are tested on made-up readings. They
have not been compared with a measured head angle.

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
