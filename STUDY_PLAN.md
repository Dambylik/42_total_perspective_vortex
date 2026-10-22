# Total Perspective Vortex — To-Do List & Study Plan

**Goal:** classify motor-imagery EEG (movement A vs B) using MNE + scikit-learn,
with your **own** dimensionality-reduction algorithm (CSP recommended) plugged
into an `sklearn.Pipeline`. Target: **≥60% mean accuracy on never-learned test data** covering the six
experiments mentioned in the subject; its example reports all 109 subjects.
The exact six-experiment definition needs to be documented (see below).

**Hard constraints from the subject (pin these to the wall):**
- Python. Use **MNE** (EEG) + **scikit-learn** (ML).
- You must **implement** the dimensionality-reduction algorithm yourself
  (CSP/PCA/ICA) as an sklearn transformer (`BaseEstimator` + `TransformerMixin`).
- The subject explicitly permits NumPy/SciPy eigenvalue, singular-value and covariance
  routines. Implement the reduction algorithm yourself; ordinary array operations
  are needed too. Do not replace your implementation with a library CSP/PCA/ICA.
- Suggested CLI with three modes, matching the subject's examples (which are not an exact-output specification):
  - `python mybci.py <subject> <run> train` → prints cross_val_score fold scores + mean
  - `python mybci.py <subject> <run> predict` → streams epochs, prints prediction vs truth, accuracy
  - `python mybci.py` → runs all 6 experiments over all 109 subjects, prints per-experiment + global mean
- Prediction runs on a simulated **stream**: predict each chunk **within 2 s** of it entering the pipeline. **Do not use mne-realtime.**
- Only the Python program goes in the repo, **not the dataset**.

**Suggested repo layout (keep it flat, ponytail):**
```
mybci.py            # CLI entry: train / predict / run-all
pipeline.py         # build_pipeline(): CSP + classifier
csp.py              # your CSP transformer (BaseEstimator, TransformerMixin)
preprocess.py       # load, filter, epoch, extract features
visualize.py        # required raw vs filtered signal plots
requirements.txt
README.md
```
`# ponytail: split further only if a file passes ~200 lines.`

---

## How we will study

For each small lesson:
1. Explain the idea in plain language, then introduce the necessary notation.
2. Predict the result of a tiny example before running it.
3. Type the code yourself, run it, and inspect values, types and shapes.
4. Explain each line in your own words; change one input and predict the effect.
5. Rebuild the small piece from memory before moving on.

The existing Python files and `RECAP.md` are reference material, not verified proof
that the project meets every requirement. Leave them available while learning;
do not copy large blocks you cannot yet explain. Recorded accuracy claims need
reproducible measurements before they can be used as project results.

## Step -1 — Python foundations, learned through this project

- NumPy arrays: shape, axis, indexing, broadcasting, transpose and matrix multiplication.
- Classes, instances, methods, `self`, inheritance and learned attributes: introduce
  these before writing the CSP transformer, not all at once on day one.

**Checkpoint:** explain the difference between one value, one channel's time series,
one multi-channel trial, and a batch of trials.

## Step 0 — Environment & dataset

**Do:**
- Create a venv, install `mne`, `scikit-learn`, `numpy`, `scipy`, `matplotlib`.
- Download the PhysioNet **EEG Motor Movement/Imagery Dataset** (EEGBCI).
  Easiest: `mne.datasets.eegbci.load_data(subject, runs)` downloads on demand
  and caches it. Choose a cache outside the repo, or explicitly ignore its directory;
  a downloader does not itself prevent committing data.
- `requirements.txt` + a one-line "how to run" in the README.

**Learn / check:**
- The dataset structure: 109 subjects, 14 runs each, `.edf` files, 64 channels, 160 Hz.
- Map run numbers to labels before coding. PhysioNet defines **four motor task
  conditions**, repeated three times, plus two baseline runs:

| Runs | Condition | T1 | T2 |
|---|---|---|---|
| 1, 2 | eyes-open / eyes-closed baseline | not motor classification | not motor classification |
| 3, 7, 11 | executed movement | left fist | right fist |
| 4, 8, 12 | imagined movement | left fist | right fist |
| 5, 9, 13 | executed movement | both fists | both feet |
| 6, 10, 14 | imagined movement | both fists | both feet |

**Unresolved subject wording:** the PDF asks for six experiment types but does not
list their definitions. Four motor conditions plus two baselines are not six
interchangeable A/B motor tasks. The existing code adds two pooled real+imagined
conditions; this is a project interpretation, not an official dataset definition.
Document that choice and resolve it against evaluation expectations before the
full benchmark. We can start learning with one unambiguous motor task now.

Dataset reference: [PhysioNet EEG Motor Movement/Imagery Dataset](https://www.physionet.org/content/eegmmidb/1.0.0/).

**Best practice:** never commit data or the venv. Add `.venv/`, `*.edf`, `mne_data/`
to `.gitignore` on day one.

---

## Step 1 — Load & explore raw EEG (Preprocessing part 1)

**Do:**
- Load one subject/run with `mne.io.read_raw_edf`.
- Read the annotations/events (`T0`=rest, `T1`, `T2`=the two movements).
- Plot raw signal (`raw.plot()`), the montage/sensor positions, and the PSD
  (`raw.compute_psd().plot()`) so you *see* the noise before filtering.

**Learn / check:**
- What an **EEG signal** is: microvolts over time, per electrode (channel).
- **Channels vs time vs events** — the shape of your data (`n_channels × n_times`).
- What **events/annotations** are and how MNE stores them.
- Electrode placement: this dataset uses the **10-10 system**, an extension of
  10-20 (start by locating C3/Cz/C4 over motor cortex —
  that's where hand/feet imagery shows up).

**Concepts to verify you understand:**
- Sampling rate (160 Hz → what frequencies are even representable? → Nyquist = 80 Hz).
- Why raw EEG looks like noise (mains hum at 50/60 Hz, drift, blinks).

**Required deliverable:** keep a runnable visualization script (e.g. `visualize.py`)
showing raw and filtered signals. Temporary experiments can be discarded, but the
subject explicitly requires this script. Inspect the data before trusting a score.

---

## Step 2 — Filter & epoch (Preprocessing part 2 + feature extraction)

**Do:**
- Band-pass filter to the motor-relevant band, **8–30 Hz** (mu + beta rhythms).
  `raw.filter(8., 30.)`. Plot the PSD again — confirm the band survived, rest is gone.
- Set the standard montage; optionally re-reference.
- Build **epochs**: cut the continuous signal into windows locked to each event
  (`mne.Epochs`). Start with a justified post-cue window, e.g. 0.5–2.5 s,
  and explicit baseline handling (`baseline=None` is a starting choice).
  Each epoch = one trial with a label. Window selection is a validation decision.
- Decide your **features**: the classic BCI move is to feed the *band-passed epochs*
  straight into CSP (CSP consumes `n_epochs × n_channels × n_times`). The subject also
  hints at band-power per channel — CSP effectively gives you that.

**Learn / check:**
- **Band-pass filtering**: what "keeping only 8–30 Hz" means and why those bands
  carry motor-imagery information (event-related desynchronization).
- **Epoching**: turning continuous data + events into a labeled 3-D array `(trials, channels, samples)`.
- Difference between **filtering** (frequency selection) and **feature extraction**
  (turning signal into numbers a classifier can use).

**Concepts to verify:**
- Filtering continuous data helps reduce epoch-edge artifacts, but does **not**
  guarantee causality. Offline zero-phase filtering uses future samples. For
  playback, define which samples are available and use consistent preprocessing.
- MNE stores EEG in volts internally; plots often display microvolts.
- What `X` and `y` look like at the end: `X = epochs.get_data()`, `y = epochs.events[:, -1]`.

**Best practice:** wrap this as `load_data(subject, runs) -> (X, y)` in `preprocess.py`.
One function, deterministic, testable. Assert the output shapes.

---

## Step 3 — Prove the pipeline works with sklearn/MNE's own CSP FIRST

**Do:**
- Build `Pipeline([('csp', mne.decoding.CSP()), ('clf', LinearDiscriminantAnalysis())])`.
- Run `cross_val_score(pipeline, X, y, cv=...)` on one subject before writing
  your own algorithm. Check labels, class balance and scores against a simple
  majority-class baseline. This is a debugging exercise, not evidence that
  the global 60% target has been met.

**Learn / check:**
- What an **sklearn Pipeline** is and why it prevents data leakage (fit only on train fold).
- What **cross_val_score** does (K-fold: train on k-1 folds, score on the held-out one).
- Pick a classifier: **LDA** is the standard first choice for CSP+EEG; SVM/LogReg also fine.

**Concepts to verify:**
- **Data leakage**: why you must fit CSP *inside* each CV fold, never on the whole set.
- The train/validation/test split diagram in the subject: CV finds params on train,
  test set is touched **once** at the end.

**Best practice:** compare on identical splits. Low reference accuracy can reflect
preprocessing, labels, a difficult subject, limited data or modeling choices; it
does not by itself identify a preprocessing bug.

---

## Step 4 — Implement your own CSP as an sklearn transformer

**Do:**
- Create `class CSP(BaseEstimator, TransformerMixin)` with `fit(X, y)` and `transform(X)`.
- **fit:** for a 2-class problem, compute the per-class **covariance matrices**,
  solve the **generalized eigenvalue problem** (`scipy.linalg.eigh(cov1, cov1+cov2)`),
  select `n_components` **in total** from both eigenvalue extremes → filter matrix `W`.
  Choose and document covariance centering/normalization and regularization.
  Re-referencing can reduce rank, so learn why the eigenproblem may need rank handling.
- **transform:** project each epoch `W.T @ X`, then compute **log-variance** of each
  component as the feature vector (this is the standard CSP output for a classifier).

**Learn / check:**
- **Covariance matrix** — what it measures, how to estimate it (`np.cov`).
- **Eigenvalues / eigenvectors** and the **generalized eigenvalue problem** — CSP finds
  spatial filters that maximize variance for one class while minimizing it for the other.
- **PCA vs CSP**: PCA is unsupervised (ignores labels); CSP uses labels to separate classes.
  The subject explicitly says CSP is better for BCIs — understand *why*.
- The sklearn transformer contract: `fit` returns `self`, `transform` returns features,
  `fit_transform` comes free from `TransformerMixin`.

**Concepts to verify:**
- Why `W.T @ X` is a linear spatial projection (the subject's `Wᵀ X = X_CSP`).
  Keeping fewer components reduces dimension; it is not an invertible full basis change.
- Why log-variance rather than raw variance (often reduces skew; Gaussian features
  are not guaranteed). Define normalization and protect against log(0).
- That your `fit` uses **only training data** (sklearn calls it per fold — trust the Pipeline).

**Best practice:** leave **one runnable check** — swap your CSP for `mne.decoding.CSP`
in the pipeline on identical splits and compare scores. Align their covariance and
feature conventions first. Similar accuracy is useful evidence, not proof of correct
math. Also check output shapes, finite features, synthetic known class structure,
and that `transform` does not refit the model.

---

## Step 5 — Wire up `train` mode

**Do:**
- `python mybci.py <subject> <run> train`: load data, build pipeline with **your** CSP,
  print the fold scores array and the mean (matches the subject's example output).
- After CV, explicitly `fit` a fresh/final pipeline on the allowed training data;
  `cross_val_score` fits clones and leaves the original unfitted. Persist the whole
  fitted pipeline (`joblib.dump`) with preprocessing settings and split metadata.
- Reserve test trials/runs before tuning; `predict` must use data excluded from training.

**Learn / check:**
- `argparse` (or `sys.argv` — this is small, `sys.argv` is fine and lazier).
- Model persistence with `joblib` and why you save the *whole fitted pipeline*, not just weights.

**Best practice:** provide the illustrated commands and readable fold/mean outputs.
The PDF presents examples, not a documented exact-output checker. `argparse` is a
useful way to validate inputs; `sys.argv` is also possible.

---

## Step 6 — Wire up `predict` mode (the streaming/2-second part)

**Do:**
- Load the saved pipeline.
- Iterate epochs one at a time to **simulate a stream** (a plain loop over epochs —
  no mne-realtime). For each chunk: call `pipeline.predict`, print
  `epoch nb: [prediction] [truth] equal?`, and ensure the call returns in **< 2 s**
  using `time.perf_counter()` for elapsed timing. Measure instead of assuming speed.
  Keep labels outside prediction; use truth only to score afterwards.
- Print final accuracy.

**Learn / check:**
- What "real-time / streaming" means here: process each chunk as it "arrives",
  don't peek at future data.
- Why the 2 s budget is generous — it's about proving online feasibility, not speed hacks.

**Best practice:** a loop can deliver chunks for playback; threads are not required.
Describe chunk availability, filtering, window assembly and timing boundaries.
Preloading a file is compatible with playback, but using future samples is not
causal processing. Distinguish window collection time from the subject's <2 s
processing deadline after a chunk enters the pipeline; report measured latency.

---

## Step 7 — `run-all` mode + hit the 60% target

**Do:**
- `python mybci.py` with no args: loop the **6 experiments × 109 subjects**, train+score
  each, print per-subject then per-experiment mean, then the global mean.
- Tune the band, `n_components` and classifier using **training/validation data only**.
  Freeze choices before final testing. Repeatedly tuning against final test accuracy
  turns the test set into validation data; a new untouched test is then needed.
- Report included subjects, exclusions with reasons, per-experiment sample counts
  and the averaging rule. Do not silently skip failures or difficult subjects.

**Learn / check:**
- How the 14 runs group into the 6 experiment tasks (define this mapping once, reuse it).
- What "never-learned data" means: the test score must come from data not used in training.
- Reading an accuracy table and spotting which experiments drag the mean down.

**Concepts to verify:**
- **Overfitting vs generalization**: high train score + low test score = overfit.
- Use multiple splits to assess variation. Record random seeds so runs can be
  reproduced; an unfixed seed is not proof of honest evaluation. Consider stratified
  splits for class balance and run-grouped splits for generalization across runs.
- Keep overlapping windows from the same trial/run together when splitting to avoid
  leakage. A Pipeline prevents leakage from fitted steps, not every possible leak.

**Best practice:** loading all 109 subjects is slow — expect long runs. Cache downloaded
data (MNE does), and test your loop on 2–3 subjects before the full sweep.

---

## Step 8 — Polish, defend, submit

**Do:**
- README: how to install, how to run each mode, what accuracy you get.
- Remove disposable scratch files; retain the required before/after visualization
  script. Confirm `.gitignore` keeps data/venv and generated models out.
- Re-read the subject and check every bullet is satisfied (the CLI formats especially).

**Prepare to explain at defense (this is graded on understanding):**
- What EEG is and how motor imagery shows up in it.
- Why band-pass 8–30 Hz.
- What CSP does, the covariance + generalized-eigenvalue math, and how it differs from PCA.
- How the sklearn Pipeline + cross_val_score avoid data leakage.
- Why your streaming loop respects the 2 s constraint.

**Best practice:** if you can't explain a line, you can't defend it. Delete borrowed code
you don't understand and rewrite it your way.

---

## Concept checklist (study these across the whole project)

| Concept | Why it matters here |
|---|---|
| EEG basics, 10-20 system | Know where/what the signal is |
| Sampling rate & Nyquist | Understand 160 Hz limits |
| Band-pass filtering, mu/beta rhythms | The core preprocessing choice |
| Epoching / event-locked windows | Turns signal into labeled trials |
| Covariance matrices | Input to CSP |
| Eigenvalues, eigenvectors, SVD, generalized eigenproblem | The CSP algorithm itself |
| PCA vs CSP (unsupervised vs supervised) | Justify your choice |
| Change of basis / projection | What `Wᵀ X` means |
| sklearn `BaseEstimator`/`TransformerMixin` contract | Required by the subject |
| Pipeline & data leakage | Correct evaluation |
| Cross-validation, K-fold | `cross_val_score` requirement |
| Train/validation/test, overfitting | The 60% must generalize |

**Recommended reading order:** MNE "MEG/EEG analysis" intro tutorial →
MNE CSP decoding example → sklearn Pipeline docs → sklearn "developing your own
estimator" guide → any BCI CSP tutorial for the math.
