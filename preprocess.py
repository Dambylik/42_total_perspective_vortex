"""Load EEG from PhysioNet, filter it, and cut it into labeled epochs (X, y)."""
import hashlib
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

DATA_PATH = "./data"   # downloaded .edf files live here (git-ignored)
LOCAL_DATA_PATH = Path(__file__).resolve().parent / "physionet.org/files/eegmmidb/1.0.0"
LOW_FREQ, HIGH_FREQ = 7.0, 30.0   # mu (8-12 Hz) + beta (13-30 Hz) = motor rhythms
TMIN, TMAX = 0.5, 2.5  # seconds after the cue; skip the first 0.5 s (eye/visual reaction)

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


def main():
    from visualize import print_heat_map, plot_class_covariances
    from csp import CSP

    # Parse process
    raw6 = load_raw(1, [6])
    data, times = raw6[:][:]
    print("\nRun 6:\nShape of the experiment", data.shape)
    print("Samples per channel :", data.shape[1])
    print(f"160 measurements per second for : {data.shape[1]/160/60:.2f} mins")
    print(f"Data from the run for the first 5 channels:\n{np.round(data[:5] * 1e6, 2)}")
    print(f"Times from the run for the first 5 channels:\n{np.round(times[:5] * 1e6, 2)}")
    print("\n_____________________________________________________________________\n")

    # Preprocess process
    preprocessed_raw = filter_raw(raw6)
    print("\nRun 6 filtered by 7-30 Hz band\n", preprocessed_raw)
    preprocessed_data, preprocessed_times = preprocessed_raw[:][:]
    print("Samples per channel :", preprocessed_data.shape[1])
    print(f"160 measurements per second for : {preprocessed_data.shape[1]/160/60:.2f} mins")
    print(f"Data from the run for the first 5 channels:\n{np.round(preprocessed_data[:5] * 1e6, 2)}")
    print(f"Times from the run for the first 5 channels:\n{np.round(preprocessed_times[:5] * 1e6, 2)}")
    print("\n_____________________________________________________________________\n")

    # Formatting process
    X, y = format_data(preprocessed_raw)
    print("\nConverted into labeled EEG windows (X, y), 1 window runs 2 seconds")
    print("X shape (epochs, channels, times):", X.shape)
    print("First epoch shape:", X[0].shape)
    with np.printoptions(precision=2, suppress=True, floatmode="fixed"):
        print("First epoch in µV:\n", X[0][0] * 1e6)
    print("Its label is:", y[0])
    # print_heat_map(X[0])
    # print("label : ", y[0])
    # assert X.ndim == 3 and X.shape[0] == len(y) and set(y.tolist()) == {1, 2}
    print("\n_____________________________________________________________________\n")

    # Connect X, y to the MNE’s CSP pipeline:
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("csp", CSP(n_components=4, log=True)),
        ("classifier", LinearDiscriminantAnalysis()),
    ])

    cv = StratifiedShuffleSplit(
        n_splits=10,
        test_size=0.2,
        random_state=42,
    )
    scores = cross_val_score(pipeline, X, y, cv=cv)
    print("Validation scores:", scores)
    print(f"Mean validation accuracy: {scores.mean():.2%}")
    

    # Find channel positions by name rather than guessing their indexes.
    channel_names = load_raw(1, [4]).ch_names
    c5_index = channel_names.index("C5")
    c6_index = channel_names.index("F5")

    c5 = X[:, c5_index, :]
    c6 = X[:, c6_index, :]

    added = c5 + c6
    subtracted = c5 - c6

    for name, signals in [("C5 + C6", added), ("C5 - C6", subtracted)]:
        variances = signals.var(axis=1) * 1e12

        left_mean = variances[y == 1].mean()
        right_mean = variances[y == 2].mean()

        print(name)
        print("  Left-imagery mean variance:", left_mean, "µV²")
        print("  Right-imagery mean variance:", right_mean, "µV²")
    # Exploration only: these are not fitted classifier parameters or test scores.
    # left_covariance = mean_class_covariance(X, y, label=1)
    # right_covariance = mean_class_covariance(X, y, label=2)
    # counts = (int(np.sum(y == 1)), int(np.sum(y == 2)))
    # print("Left-imagery epochs:", counts[0])
    # print("Right-imagery epochs:", counts[1])
    # print("Mean covariance shape per class:", left_covariance.shape)
    # plot_class_covariances(left_covariance, right_covariance, run=4, counts=counts)


if __name__ == "__main__":
    main()
