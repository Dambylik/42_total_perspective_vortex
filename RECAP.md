# Total Perspective Vortex — Recap of the implementation

A Brain-Computer Interface (BCI): from a person's EEG, guess which of two
movements (A or B) they did or imagined.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python visualize.py 1 4          # see raw vs filtered EEG (subject 1, run 4)
python mybci.py 4 14 train       # cross-validation scores + save model
python mybci.py 4 14 predict     # replay run 14 as a stream, predict each epoch
python mybci.py                  # 6 experiments x 109 subjects (long: downloads ~3 GB)

python csp.py                    # self-check of our CSP on fake data
python preprocess.py             # self-check of the preprocessing shapes
```

## Files

| File | Role |
|---|---|
| `preprocess.py` | load `.edf` files → filter 7–30 Hz → cut epochs → `X, y` |
| `csp.py` | **our own** CSP, a scikit-learn transformer |
| `mybci.py` | CLI: `train`, `predict`, run-all |
| `visualize.py` | plots before/after filtering |

Flat layout, one file per step of the pipeline. No packages, no classes except
the one the subject requires (CSP). Split a file only when it gets too big.

---

## The whole pipeline in one picture

```
.edf files ─► Raw (64 channels × time) ─► band-pass 7–30 Hz ─► Epochs (trials × 64 × 321)
                                                                   │
                                  ┌────────── sklearn Pipeline ────┴──────────┐
                                  │  CSP: 64 channels → 4 log-variance features │
                                  │  LDA: 4 features → class 1 or 2             │
                                  └─────────────────────────────────────────────┘
```

---

## 1. The data (PhysioNet EEG Motor Movement/Imagery)

- 109 subjects, 14 runs each, 64 electrodes, **160 Hz** sampling rate.
- Runs 1–2: baseline (eyes open/closed) → not used.
- Runs 3–14: tasks. Each run has events **T0** (rest), **T1**, **T2** (the two movements).

| Runs | Task | T1 | T2 |
|---|---|---|---|
| 3, 7, 11 | real movement | left fist | right fist |
| 4, 8, 12 | imagined | left fist | right fist |
| 5, 9, 13 | real movement | both fists | both feet |
| 6, 10, 14 | imagined | both fists | both feet |

**Our 6 experiments** (`EXPERIMENTS` in `preprocess.py`): the 4 rows above, plus
real+imagined hands and real+imagined fists/feet.

**Why `eegbci.load_data`:** it downloads once and caches in `./data`, so the
dataset never goes in git (subject rule). `data/` and `models/` are in `.gitignore`.

## 2. Preprocessing — `preprocess.py`

### Band-pass filter 7–30 Hz
- When you move (or imagine moving), the **mu (8–12 Hz)** and **beta (13–30 Hz)**
  rhythms over the motor cortex get **weaker** on the opposite side of the brain
  (this is called *event-related desynchronization*, ERD).
  Left hand → right hemisphere (C4) gets quieter, and the other way round.
- Everything else is noise for us: slow drift (< 1 Hz), eye blinks (low freq),
  muscle (high freq), mains power (60 Hz in the USA).
- Nyquist: at 160 Hz we can only see frequencies up to 80 Hz.

**Best practice:** filter the **continuous** signal **before** cutting epochs.
Filtering short windows creates edge artifacts.

### Epochs
- An epoch = a window of signal locked on an event: `[cue + 0.5 s, cue + 2.5 s]`.
- We skip the first 0.5 s: that part is the brain reacting to the cue on screen,
  not the movement itself. The movement effect is strongest a bit later.
- `baseline=None`: CSP uses variance, so subtracting a mean changes nothing.
- Result: `X` of shape `(n_epochs, 64, 321)` and `y` in `{1, 2}`.

### Features
The subject suggests "power of the signal by frequency and by channel". After a
7–30 Hz filter, **variance = power in that band**. CSP computes the variance of
smart combinations of channels, so it gives exactly this kind of feature.

## 3. Dimensionality reduction — CSP (`csp.py`)

### Why CSP and not PCA?
| PCA | CSP |
|---|---|
| unsupervised (ignores labels) | **supervised** (uses labels) |
| finds directions with the most variance **overall** | finds directions where variance is **different between the 2 classes** |
| the biggest variance in EEG is often noise (blinks) | built exactly for "which area is more/less active" |

CSP is the standard method for motor-imagery BCIs.

### The math, step by step
1. For each epoch `E` (channels × times), compute the covariance `E Eᵀ` (channels × channels)
   and divide it by its trace. Dividing by the trace means one very loud trial
   can't dominate the average.
2. Average per class → `C_a`, `C_b`.
3. Solve the **generalized eigenvalue problem** `C_a w = λ (C_a + C_b) w`
   (`scipy.linalg.eigh`, allowed by the subject).
   - λ ∈ [0, 1] is the share of the variance along `w` that comes from class a.
   - λ ≈ 1 → strong for class a, weak for class b. λ ≈ 0 → the opposite.
4. Keep the eigenvectors with the **most extreme** λ (both ends). They form `W`.
5. `transform`: `X_CSP = Wᵀ X` (the subject's formula, a change of basis:
   64 electrodes → 4 "virtual electrodes"), then **log(normalized variance)**
   of each one → 4 numbers per epoch.

**Why log?** Variances are skewed (always positive, long tail). The log makes them
roughly Gaussian, which is exactly what LDA assumes.

### sklearn contract (`BaseEstimator` + `TransformerMixin`)
- `__init__` only stores parameters (so `clone()` and `get_params()` work in CV).
- `fit(X, y)` learns things, stores them with a trailing underscore (`filters_`), returns `self`.
- `transform(X)` uses what was learned. `fit_transform` comes free from `TransformerMixin`.

## 4. Classifier — LDA

Linear Discriminant Analysis: draws the best straight line between two Gaussian
clouds. We chose it because:
- it is the standard partner of CSP in the BCI literature,
- 4 features and ~30–90 training trials: a simple model **overfits less**,
- it is fast (predict in microseconds → the 2 s limit is easy).

Alternatives: logistic regression, SVM. Try them only if LDA isn't enough.

## 5. Pipeline + validation (no data leakage)

```python
Pipeline([("csp", CSP(n_components=4)), ("lda", LinearDiscriminantAnalysis())])
```

**Data leakage** = information from the test data getting into training. CSP
*learns* from labels, so if you fit it on all data and then cross-validate
only the LDA, the score is a lie. With a `Pipeline`, `cross_val_score` re-fits
**CSP and LDA together on each training fold only**.

Our split (see the subject's train/validation/test picture):

```
experiment runs, e.g. [6, 10, 14]
 ├── training runs [6, 10] ──► cross_val_score (10 random 80/20 splits) = validation
 │                            then fit the final model on all of them
 └── test run [14] ─────────► never seen during training = final test ("predict")
```

- **Why hold out a whole run** and not random epochs? Epochs from the same run
  are recorded minutes apart and look alike. Testing on a different run is
  harder and more honest: it's how a real BCI is used (train today, use later).
- **Why `ShuffleSplit` without `random_state`?** The subject asks for "different
  splits each time", so a good score isn't just one lucky split.

## 6. "Real-time" prediction — `predict`

- We replay the test run: a `for` loop hands one epoch at a time to the
  pipeline, as if it had just arrived. We don't use `mne-realtime` (forbidden).
- Each `predict` is timed. If it takes more than 2 s we print a warning.
  In practice it takes about 1 ms.
- **Known simplification:** we filter the whole run first. A real live system
  can't see the future, so it would use a *causal* filter (IIR,
  `mne.filter.filter_data(..., method="iir")`) on each chunk as it arrives.

## 7. Best practices used here

- **Check the pipeline with a known-good tool first**: we compared our CSP with
  `mne.decoding.CSP` on the same data. The scores should be about the same.
  Result on subjects 1–6, all 6 experiments: **ours 66.5 %**, MNE 67.6 %.
  If they're not, the bug is in your CSP; if both are bad, the bug is in preprocessing.
- **One small runnable check per tricky part**: `python csp.py` (fake data with
  a known answer) and `python preprocess.py` (shapes).
- **Constants at the top** of the file (`LOW_FREQ`, `TMIN`…) = the knobs you tune.
- **Look at the data** (`visualize.py`) before trusting any number.
- Never commit data, models or the venv.

## 8. Tuning knobs (if accuracy is too low)

| Knob | Where | Typical values |
|---|---|---|
| frequency band | `LOW_FREQ`, `HIGH_FREQ` | 7–30, 8–30, 8–13 |
| epoch window | `TMIN`, `TMAX` | 0.5–2.5, 1–2, 0–4 |
| CSP components | `CSP(n_components=...)` | 2, 4, 6, 8 |
| classifier | `build_pipeline()` | LDA, `LDA(solver="lsqr", shrinkage="auto")`, LogisticRegression |

## 9. Questions to prepare for the defense

- What is EEG, and why does imagining a movement change the 8–30 Hz power?
- Why filter before epoching?
- Explain the covariance, the generalized eigenvalue problem, and `Wᵀ X`.
- CSP vs PCA: why is supervised better here?
- Why does the Pipeline prevent leakage? What would happen without it?
- Why is the test run "never-learned data"?
- How do you guarantee the 2 s limit?

## Bonus ideas (not done)
- Wavelet / filter-bank features (FBCSP: CSP on several frequency bands).
- Your own classifier (e.g. LDA from scratch: class means + shared covariance).
- Your own eigen-decomposition (hard: noisy data, needs regularization).
