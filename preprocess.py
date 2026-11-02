"""Load EEG from PhysioNet, filter it, and cut it into labeled epochs (X, y)."""
import hashlib
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

DATA_PATH = "./data"
LOCAL_DATA_PATH = Path(__file__).resolve().parent / "physionet.org/files/eegmmidb/1.0.0"
LOW_FREQ, HIGH_FREQ = 7.0, 30.0 
TMIN, TMAX = 0.5, 2.5

EXPERIMENTS = [
    [3, 7, 11],               # 0: real movement,     left fist vs right fist
    [4, 8, 12],               # 1: imagined movement, left fist vs right fist
    [5, 9, 13],               # 2: real movement,     both fists vs both feet
    [6, 10, 14],              # 3: imagined movement, both fists vs both feet
    [3, 7, 11, 4, 8, 12],     # 4: real + imagined,   left fist vs right fist
    [5, 9, 13, 6, 10, 14],    # 5: real + imagined,   both fists vs both feet
]

mne.set_log_level("ERROR")


def find_experiment(run):
    """Return the runs of the first experiment that contains `run`."""
    for runs in EXPERIMENTS:
        if run in runs:
            return runs
    raise ValueError(f"run {run} is not a motor task (use a run between 3 and 14)")


def load_raw(subject, runs):
    """Prefer wget recordings; use MNE's cache/downloader for missing files."""
    if isinstance(runs, (int, np.integer)): # if One integer → convert to [6], if [6, 10] is already a list → leave it unchanged
        runs = [runs] # why, cause t can loop over [6] or [6, 10], but cannot loop over the integer 6.

    #This block only loads the expected fingerprints. It does not check the recordings yet. Later code calculates each local recording’s fingerprint and compares it with the expected one.
    checksums = {}
    manifest = LOCAL_DATA_PATH / "SHA256SUMS.txt"
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            digest, filename = line.split(maxsplit=1)
            checksums[filename.lstrip("*")] = digest

    files = []
    for run in runs: # cause we need here a list
        relative = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf" #S004/S004R06.edf
        local_file = LOCAL_DATA_PATH / relative # Joins that relative filename to the dataset folder, producing the full file path.
        use_local = local_file.is_file() # use_local becomes True if the file exists, otherwise False.
        if use_local and relative in checksums:
            with local_file.open("rb") as stream: #Opens the file for reading binary data. EDF recordings are binary files. with ensures the file closes afterward.
                digest = hashlib.file_digest(stream, "sha256").hexdigest() #Reads the file and calculates its SHA-256 fingerprint. hexdigest() represents that fingerprint as text.
            use_local = digest == checksums[relative] #Compares the calculated fingerprint with the expected fingerprint:
            if not use_local:
                print(f"Incomplete or invalid local recording: {local_file}; using MNE instead.")
        if use_local:
            files.append(str(local_file))
        else:
            files.extend(eegbci.load_data(subject, [run], path=DATA_PATH, update_path=False))

    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)  # "Fc5." -> "FC5"
    raw.set_montage("standard_1005")
    return raw


def filter_raw(raw):
    """Keep only the 7-30 Hz band. Returns a new Raw (the original is untouched)."""
    return raw.copy().filter(LOW_FREQ, HIGH_FREQ, picks="eeg")


def format_data(raw):
    """Convert a filtered recording into labeled EEG windows (X, y)."""
    # Drop rest (T0); retain the two task labels.
    events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2})
    epochs = mne.Epochs(raw, events, tmin=TMIN, tmax=TMAX, picks="eeg",
                        baseline=None, preload=True)
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]
    return X, y


def get_epochs(subject, runs):
    """Read, filter and format recordings; shared by evaluation and exploration."""
    raw = load_raw(subject, runs)
    filtered = filter_raw(raw)
    return format_data(filtered)


def mean_class_covariance(X, y, label):
    """Average the epoch covariance matrices belonging to one task label."""
    class_epochs = X[y == label]
    if len(class_epochs) == 0:
        raise ValueError(f"No epochs found for label {label}")
    covariances = [np.cov(epoch) for epoch in class_epochs]
    return np.mean(covariances, axis=0)
