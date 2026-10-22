# Total Perspective Vortex

A Python learning project that uses EEG recordings to distinguish two movement tasks, such as imagining the left fist versus the right fist.

The aim is to understand the complete journey from recorded electrical measurements to a tested prediction: read the data with **MNE**, filter it, cut it into labeled windows, implement **Common Spatial Patterns (CSP)** ourselves, and connect CSP to a classifier using a **scikit-learn Pipeline**.

This is classification of predefined experimental tasks, not unrestricted mind reading. An electrode does not record a sentence such as “move my left hand.” We look for statistical patterns associated with the task instructions.

## Start here

Run commands from this repository's directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Explore an imagined left/right-fist recording; display and save plots.
python visualize.py 1 4

# Explore synthetic waves with interactive controls; no download required.
python sampling_demo.py

# Save both synthetic demonstrations as PNG images.
python sampling_demo.py --save docs/images/sampling_demo.png

# Train on related recordings, reserving run 14 for later testing.
python mybci.py 4 14 train
python mybci.py 4 14 predict
```

The first download can take time before a plot appears. Recordings are cached under `data/`; later uses of the same recording read the cache. Dataset files, saved models and the virtual environment are ignored by Git.

`python mybci.py` starts the full configured evaluation across 109 participants and six groups. Start with one participant and task before attempting this larger download and computation. See the experiment-definition and evaluation limitations below.

## What does EEG measure?

**EEG** stands for electroencephalography. Electrodes on the scalp measure tiny electrical potential differences over time. One recorded stream is called a **channel**.

Imagine microphones placed around an orchestra. Each microphone hears a mixture of sources. Likewise, each EEG channel contains mixed activity, including unwanted contributions from eyes, muscles and electrical equipment. One channel is not one isolated neuron or one isolated brain area.

![Lecture slide listing non-invasive and invasive brain-recording methods](<docs/images/Screenshot 2026-10-22 at 16.09.22.png>)

*Different recording methods. This project uses scalp EEG, in the non-invasive group. The other methods provide context; they are not inputs to this program. ECoG is a type of intracranial recording, not a synonym for every form of intracranial EEG.*

![Lecture comparison of the spatial and temporal scales of brain-research methods](<docs/images/Screenshot 2026-10-22 at 16.09.04.png>)

*Read the horizontal axis as time scale and the vertical axis as spatial size. The EEG/MEG region illustrates sensitivity to rapid changes over relatively broad spatial scales. This is a historical overview, not an exact accuracy specification for our recording equipment. The project uses changes over time and across channels; it does not locate individual neurons.*

## Participants, runs and labels

A **subject** is one participant. A **run** is one recording session for that participant. A run contains multiple instructed task periods and rests, rather than just one movement.

The PhysioNet dataset has 109 participants, 64 EEG channels and 14 runs per participant. The researchers defined this schedule:

| Run numbers | Condition | T1 | T2 |
|---|---|---|---|
| 1 | Eyes-open resting baseline | Not a motor classification task | Not a motor classification task |
| 2 | Eyes-closed resting baseline | Not a motor classification task | Not a motor classification task |
| 3, 7, 11 | Actual movement | Left fist | Right fist |
| 4, 8, 12 | Imagined movement | Left fist | Right fist |
| 5, 9, 13 | Actual movement | Both fists | Both feet |
| 6, 10, 14 | Imagined movement | Both fists | Both feet |

**T0 means rest. T1 and T2 require the run number to interpret them.** For example, run 3 + T1 means actual left-fist movement, while run 6 + T1 means imagined movement of both fists. These annotations record the requested task; they are not predictions made by our model.

Source: [PhysioNet experimental protocol and annotation definitions](https://archive.physionet.org/physiobank/database/eegmmidb/).

### Why is `EXPERIMENTS` a list of lists?

Each inner list specifies recordings used together for one classification problem. For example, `[3, 7, 11]` groups three repetitions of actual left/right-fist movement. We can learn from two runs and test on the third.

The first four groups in `preprocess.py` follow the four motor conditions above. The last two groups pool actual and imagined recordings with matching movement labels:

```python
[3, 7, 11, 4, 8, 12]   # left fist versus right fist, pooled conditions
[5, 9, 13, 6, 10, 14]  # both fists versus both feet, pooled conditions
```

**These two pooled groups are implementation choices.** The assignment mentions six experiments but does not define this exact mapping. Baseline runs are not two additional interchangeable A/B motor tasks.

## Sampling frequency is not signal frequency

A sampling frequency of **160 Hz** means each channel supplies 160 measurements per second. It does not mean the EEG completes 160 wave cycles per second.

| Quantity | Meaning |
|---|---|
| Sampling frequency: 160 Hz | Take 160 measurements each second, per channel |
| Signal frequency: 10 Hz | A wave repeats 10 times each second |
| Voltage amplitude: µV | The size of a voltage fluctuation; 1 µV = 0.000001 V |
| Power density: µV²/Hz | How signal power is distributed across frequencies |

Measurement times are spaced by `1 / 160 = 0.00625` seconds. The measurement **values** come from the recording equipment; the sampling rate does not determine them.

![Synthetic 10 Hz wave with 160 measurement dots during one second](<docs/images/sampling_demo.png>)

*Blue is an invented continuous wave; orange dots show where measurements are taken. A 10 Hz wave sampled at 160 Hz has 16 samples per cycle. In `sampling_demo.py`, changing the wave frequency does not change the sampling rate. The initial amplitudes and phases are chosen for illustration, not measured from EEG.*

## One channel can contain several frequencies

A musical chord has several notes, but one microphone records their combined sound. Similarly, one EEG channel contains mixed frequency contributions.

![Synthetic component waves, their combined signal and an ideal filtered result](<docs/images/sampling_demo_mixture.png>)

*Top: three invented ingredient waves. Middle: add their values at every sample time to create one channel. Bottom: ideal 7–30 Hz filtering retains the 10 Hz ingredient. The demonstration reconstructs the selected ingredients for clarity; it is not the real MNE filtering algorithm.*

At a given instant, a contribution of +10 µV and another of −4 µV add to +6 µV. Contributions sometimes reinforce each other and sometimes cancel. In the real dataset the equipment records the mixture directly; it does not supply separate arrays for each frequency.

The demonstration waves start at zero because they were generated using sine functions with zero starting phase. Real EEG has no requirement to start at zero or to align different frequency components.

## Why filter to approximately 7–30 Hz?

![Lecture slide listing conventional EEG frequency bands and discussing mu and beta activity](<docs/images/Screenshot 2026-10-22 at 16.37.49.png>)

*The connection relevant here is that mu and beta activity can weaken during movement or movement imagery. Band boundaries vary between conventions and individuals. The slide's broad associations are context, not rules that identify a thought or diagnose a state from one peak.*

Our starting band includes **mu**, around 8–13 Hz over sensorimotor regions, and **beta**, roughly 13–30 Hz. Changes in their power can help distinguish motor tasks. Mu overlaps the frequency range often called alpha, but location and context matter: every peak near 10 Hz is not automatically motor-related mu.

The current code is:

```python
raw.copy().filter(7.0, 30.0, picks="eeg")
```

It filters each EEG channel separately. It does not combine the 64 channels or change the sampling rate. It calculates new signal values to reduce frequency contributions outside the chosen band. A real filter has transition regions, rather than perfect cutoff walls.

**Filtering changes the signal; zooming changes only the view.** Other frequency bands are not universally useless, and this filter does not remove every artifact. The 7–30 Hz choice is a starting configuration, also used in [MNE's motor-imagery CSP example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html), to assess through validation.

## Read the five generated plots

`visualize.py` produces these files in `docs/images/`, relative to the script location. Running it again replaces the same filenames. The shown task descriptions depend on the selected run.

| File | Question it answers |
|---|---|
| `sensor_positions.png` | Where are the channels located on the head? |
| `signal_before.png` | How do original voltage measurements change over time? |
| `spectrum_before.png` | Which frequencies contribute power before filtering? |
| `signal_after.png` | How does the voltage signal look after filtering? |
| `spectrum_after.png` | Which frequency contributions were reduced? |

### Sensor locations

![Standard sensor positions matched to the recording's channel names](<docs/images/sensor_positions.png>)

The head is viewed from above, with the nose at the top. The dataset follows the **10–10 placement system**, using proportional head measurements to identify electrode locations. `raw.set_montage("standard_1005")` attaches positions from MNE's more extensive built-in template by channel name. It does not add sensors or alter signal values. These are standard positions, not individually measured head coordinates.

### Signals before and after filtering

![Original time signals with task annotations](<docs/images/signal_before.png>)

![Filtered time signals with the same annotation meanings](<docs/images/signal_after.png>)

Each trace is one channel. Traces are stacked for readability; their vertical placement is not a comparison of absolute voltage between channels. The upper time axis shows the enlarged section, initially approximately 0–10 seconds. The bottom timeline shows the whole recording; its selection box indicates the enlarged section.

For default run 4, blue T0 means rest, orange T1 means imagined left fist, and green T2 means imagined right fist. Colours represent known annotations, not model predictions.

Both plots now use `scalings={"eeg": 100e-6}` so amplitudes can be compared using the same display ruler. MNE's full marker is 200 µV with this setting. It is a voltage ruler, not standardized power, a maximum value, or a restriction on the recorded data. Display scaling does not change the data.

## Spectra before and after filtering

### Why does our spectrum stop at 80 Hz?

Sampling must be sufficiently fast relative to the highest signal frequency.
Half the sampling rate is the **Nyquist frequency**: `160 / 2 = 80 Hz`. 
Frequencies must be below this boundary to be represented unambiguously under the sampling assumptions. Faster activity can masquerade as slower activity, called **aliasing**. Think of a filmed wheel appearing to rotate backwards when the camera captures too few frames.

![Lecture slide introducing the Nyquist-Shannon sampling theorem](<docs/images/Screenshot 2026-10-22 at 16.53.03.png>)

![Power spectral density before filtering](<docs/images/spectrum_before.png>)

![Power spectral density after 7–30 Hz filtering](<docs/images/spectrum_after.png>)

Each coloured curve represents one channel. All curves share the same frequency and power axes. A curve below others at 10 Hz has less estimated power density there; it does not use a different frequency scale. Curves can hide behind one another where they overlap.

The vertical axis uses decibels relative to a reference. Negative decibel values do not mean negative physical power. After filtering, the low-frequency region and frequencies above the upper transition region are much weaker; the retained band remains prominent. The spectrum is an estimate, not a list of perfectly isolated biological sources.

`visualize.py` also prints the C3 spectrum between 7 and 30 Hz in linear µV²/Hz. Its strongest-frequency output is a peak in that selected range, summarizing the original recording across rests and tasks. It is not a timestamp, peak voltage amplitude, or proof of a particular movement. To investigate task differences, compare labeled windows instead of relying on one whole-recording peak.

## From a recording to a prediction

```text
EDF files
  → MNE Raw: channels × samples
  → band-pass filtering, separately per channel
  → labeled epochs: trials × channels × samples
  → custom CSP: spatial combinations and log-variance features
  → LDA classifier: predicted class
```

**Epoching:** `get_epochs()` selects T1/T2 events and extracts windows from 0.5 to 2.5 seconds after each onset. At 160 Hz, MNE's inclusive endpoints normally produce 321 samples for that window. `X` contains those signal windows; `y` contains the associated labels. T0 rest is excluded from this binary classifier.

**CSP:** filtering selected frequencies within each channel. CSP subsequently learns weighted combinations **across channels**, using the training labels. It looks for combinations whose variance differs between the classes. Covariance summarizes how channels vary together; an eigenvalue calculation finds useful projection directions. The current transformer reduces each epoch to four log-normalized-variance features. These are discriminative measurements, not direct proof that a particular brain area is active.

**LDA:** Linear Discriminant Analysis uses those features to choose a class. Picture two clouds of feature points and a linear boundary used to separate them.

**Pipeline:** CSP and LDA are fitted together inside each training split. Fitting CSP on all examples before evaluating would leak validation/test information into the learned features.

## Training, validation and final testing

For `python mybci.py 4 14 train`, the current code selects the related group `[6, 10, 14]`, trains using runs 6 and 10, and reserves run 14 for `predict`.

Within training data, `cross_val_score` uses `ShuffleSplit`: ten random 80/20 training/validation splits. This is repeated random holdout, not five fixed folds. A **fold** means a subset of examples, not an experiment or necessarily a whole recording. Validation helps select settings; final test results must not guide repeated tuning.

After cross-validation, `pipeline.fit(X, y)` fits the saved model on all allowed training examples. Cross-validation fits separate clones; it does not fit this original pipeline for later prediction.

The assignment's goal is at least 60% mean test accuracy under its required evaluation. This README does not claim a verified full-dataset result.

### Current limitations to understand before final evaluation

- The two pooled experiment definitions require clarification against assignment expectations.
- `run_all()` holds out the last run of each configured group and averages subject accuracies within each group, then averages the six group means. It does not perform the training-mode cross-validation inside that sweep.
- `run_all()` catches all exceptions and skips the failed subject/group pair. A final report needs exclusions and coverage, and should distinguish data problems from programming errors.
- Training-mode random splits currently have no fixed seed or explicit stratification. Recording seeds and considering class balance/run grouping will improve evaluation clarity.
- Playback supplies precomputed epochs one at a time and times prediction. Filtering the complete recording beforehand can use future samples; this does not establish causal live processing or the latency of the complete preprocessing chain. The assignment's two-second deadline must be assessed with explicit timing boundaries.

## Code map and learning path

| File | Role |
|---|---|
| `preprocess.py` | Download/read recordings, match channel names and locations, filter, create epochs |
| `visualize.py` | Inspect metadata, annotations, before/after plots and C3 power density |
| `csp.py` | Custom scikit-learn-compatible CSP transformer |
| `mybci.py` | Training, prediction and configured evaluation commands |
| `mybci_2.py` | Work-in-progress rewrite for learning |
| `inspect_data.py` | Small inspection script for signal slices and annotations |
| `sampling_demo.py` | Interactive synthetic sampling and mixed-frequency illustrations |
| `STUDY_PLAN.md` | Learning sequence and requirement checkpoints |
| `my_notes.md` | Personal concept notes |
| `RECAP.md` | Earlier implementation recap; reported results need reproducible verification |

Learn in this order: identify participant/run/labels → inspect channel values and times → understand sampling versus frequency → read spectra → filter and compare → create epochs → understand covariance and CSP → build the classifier → evaluate on unseen data. Type small sections yourself and explain their inputs, outputs and purpose before moving on.

## Sources and screenshot attribution

- [PhysioNet EEG Motor Movement/Imagery experimental protocol](https://archive.physionet.org/physiobank/database/eegmmidb/): recording schedule and annotation meanings.
- [MNE motor-imagery CSP example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html): reference implementation workflow.
- [MNE montage documentation for version 1.12](https://mne.tools/1.12/generated/mne.channels.make_standard_montage.html): built-in electrode templates used by the installed project version.
- The four lecture screenshots were supplied by the repository owner from a YouTube video. UC San Diego branding is visible on the oscillation and sampling slides. The exact video URL, speaker and original source of the historical comparison figure were not supplied, so no more specific attribution is asserted. Original slide content belongs to its respective creators.
- The PNG files named `signal_*`, `spectrum_*` and `sensor_positions` are local recording visualizations. The `sampling_demo*` images are synthetic teaching examples, not measured EEG results.
