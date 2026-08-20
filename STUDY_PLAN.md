# Total Perspective Vortex — To-Do List & Study Plan

**Goal:** classify motor-imagery EEG (movement A vs B) using MNE + scikit-learn,
with your **own** dimensionality-reduction algorithm (CSP recommended) plugged
into an `sklearn.Pipeline`. Must reach **≥60% mean accuracy** across 6 experiments
and 109 subjects.

**Hard constraints from the subject (pin these to the wall):**
- Python. Use **MNE** (EEG) + **scikit-learn** (ML).
- You must **implement** the dimensionality-reduction algorithm yourself
  (CSP/PCA/ICA) as an sklearn transformer (`BaseEstimator` + `TransformerMixin`).
- Numpy/scipy allowed **only** for eigenvalues, SVD, covariance estimation.
- One CLI, three modes:
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
requirements.txt
README.md
```
`# ponytail: split further only if a file passes ~200 lines.`

---

## Step 0 — Environment & dataset

**Do:**
- Create a venv, install `mne`, `scikit-learn`, `numpy`, `scipy`, `matplotlib`.
- Download the PhysioNet **EEG Motor Movement/Imagery Dataset** (EEGBCI).
  Easiest: `mne.datasets.eegbci.load_data(subject, runs)` downloads on demand
  and caches it — dataset stays out of your repo automatically.
- `requirements.txt` + a one-line "how to run" in the README.

**Learn / check:**
- What a venv is and why (`python -m venv`, activate, `pip freeze`).
- The dataset structure: 109 subjects, 14 runs each, `.edf` files, 64 channels, 160 Hz.
- The **6 experiment types** (the 6 "tasks" you'll score): they come from grouping
  the runs — e.g. real vs imagined hand/feet movement. Map run numbers → task before coding.

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
- The **10-20 electrode system** (just enough to know C3/Cz/C4 sit over motor cortex —
  that's where hand/feet imagery shows up).

**Concepts to verify you understand:**
- Sampling rate (160 Hz → what frequencies are even representable? → Nyquist = 80 Hz).
- Why raw EEG looks like noise (mains hum at 50/60 Hz, drift, blinks).

**Best practice:** write a small `explore.py` you throw away later — exploration code
is not production code. Look at the data with your eyes before trusting any number.

---

## Step 2 — Filter & epoch (Preprocessing part 2 + feature extraction)

**Do:**
- Band-pass filter to the motor-relevant band, **8–30 Hz** (mu + beta rhythms).
  `raw.filter(8., 30.)`. Plot the PSD again — confirm the band survived, rest is gone.
- Set the standard montage; optionally re-reference.
- Build **epochs**: cut the continuous signal into windows locked to each event
  (`mne.Epochs`, e.g. `tmin=-0.2`, `tmax=4.0`). Each epoch = one trial with a label.
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
- Why filter *before* epoching (edge effects / causality).
- What `X` and `y` look like at the end: `X = epochs.get_data()`, `y = epochs.events[:, -1]`.

**Best practice:** wrap this as `load_data(subject, runs) -> (X, y)` in `preprocess.py`.
One function, deterministic, testable. Assert the output shapes.

---

## Step 3 — Prove the pipeline works with sklearn/MNE's own CSP FIRST

**Do:**
- Build `Pipeline([('csp', mne.decoding.CSP()), ('clf', LinearDiscriminantAnalysis())])`.
- Run `cross_val_score(pipeline, X, y, cv=...)`. Get a number ≥60% on one subject
  **before** writing any of your own algorithm.

**Learn / check:**
- What an **sklearn Pipeline** is and why it prevents data leakage (fit only on train fold).
- What **cross_val_score** does (K-fold: train on k-1 folds, score on the held-out one).
- Pick a classifier: **LDA** is the standard first choice for CSP+EEG; SVM/LogReg also fine.

**Concepts to verify:**
- **Data leakage**: why you must fit CSP *inside* each CV fold, never on the whole set.
- The train/validation/test split diagram in the subject: CV finds params on train,
  test set is touched **once** at the end.

**Best practice:** this de-risks everything. If the reference CSP can't hit 60%, your
preprocessing is the problem — fix that before blaming your own implementation.
`# ponytail: don't build your own algorithm until the borrowed one proves the target is reachable.`

---

## Step 4 — Implement your own CSP as an sklearn transformer

**Do:**
- Create `class CSP(BaseEstimator, TransformerMixin)` with `fit(X, y)` and `transform(X)`.
- **fit:** for a 2-class problem, compute the per-class **covariance matrices**,
  solve the **generalized eigenvalue problem** (`scipy.linalg.eigh(cov1, cov1+cov2)`),
  sort eigenvectors by eigenvalue, keep the top/bottom `n_components` → filter matrix `W`.
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
- Why `W.T @ X` is a "change of basis" / projection (the subject's `Wᵀ X = X_CSP`).
- Why log-variance, not raw variance (makes features more Gaussian → LDA likes it).
- That your `fit` uses **only training data** (sklearn calls it per fold — trust the Pipeline).

**Best practice:** leave **one runnable check** — swap your CSP for `mne.decoding.CSP`
in the pipeline and confirm `cross_val_score` is within a few % of the reference.
That's your proof the math is right. `# ponytail: one assert against the reference beats a test suite here.`

---

## Step 5 — Wire up `train` mode

**Do:**
- `python mybci.py <subject> <run> train`: load data, build pipeline with **your** CSP,
  print the fold scores array and the mean (matches the subject's example output).
- Persist the fitted pipeline (`joblib.dump`) so `predict` can reload it.

**Learn / check:**
- `argparse` (or `sys.argv` — this is small, `sys.argv` is fine and lazier).
- Model persistence with `joblib` and why you save the *whole fitted pipeline*, not just weights.

**Best practice:** match the subject's printed format exactly — the evaluator compares
against it. `# ponytail: sys.argv over argparse for 3 fixed args.`

---

## Step 6 — Wire up `predict` mode (the streaming/2-second part)

**Do:**
- Load the saved pipeline.
- Iterate epochs one at a time to **simulate a stream** (a plain loop over epochs —
  no mne-realtime). For each chunk: call `pipeline.predict`, print
  `epoch nb: [prediction] [truth] equal?`, and ensure the call returns in **< 2 s**
  (it will easily — just `time.time()` around it to prove it and print/assert).
- Print final accuracy.

**Learn / check:**
- What "real-time / streaming" means here: process each chunk as it "arrives",
  don't peek at future data.
- Why the 2 s budget is generous — it's about proving online feasibility, not speed hacks.

**Best practice:** the "stream" is just a `for` loop over already-loaded epochs with a
timing check. Don't build a threading/queue system for it.
`# ponytail: a for-loop with a time.time() guard IS the stream simulator; skip queues/threads.`

---

## Step 7 — `run-all` mode + hit the 60% target

**Do:**
- `python mybci.py` with no args: loop the **6 experiments × 109 subjects**, train+score
  each, print per-subject then per-experiment mean, then the global mean.
- Tune the band (8–30 vs 8–12/13–30), `n_components`, and classifier until mean ≥ 60%.

**Learn / check:**
- How the 14 runs group into the 6 experiment tasks (define this mapping once, reuse it).
- What "never-learned data" means: the test score must come from data not used in training.
- Reading an accuracy table and spotting which experiments drag the mean down.

**Concepts to verify:**
- **Overfitting vs generalization**: high train score + low test score = overfit.
- Why the subject demands *different splits each time* (shuffle CV) — to prove you're not
  memorizing one lucky split.

**Best practice:** loading all 109 subjects is slow — expect long runs. Cache downloaded
data (MNE does), and test your loop on 2–3 subjects before the full sweep.

---

## Step 8 — Polish, defend, submit

**Do:**
- README: how to install, how to run each mode, what accuracy you get.
- Remove throwaway exploration scripts. Confirm `.gitignore` keeps data/venv out.
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
