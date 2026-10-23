"""Load EEG from PhysioNet, filter it, and cut it into labeled epochs (X, y)."""
import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

DATA_PATH = "./data"   # downloaded .edf files live here (git-ignored)
LOW_FREQ, HIGH_FREQ = 7.0, 30.0   # mu (8-12 Hz) + beta (13-30 Hz) = motor rhythms
TMIN, TMAX = 0.5, 2.5  # seconds after the cue; skip the first 0.5 s (eye/visual reaction)

# The 6 experiments = groups of runs of the same task (PhysioNet run numbers).
# In every group, T1 and T2 are the two classes we try to tell apart.
EXPERIMENTS = [
    [3, 7, 11],               # 0: real movement,     left fist vs right fist
    [4, 8, 12],               # 1: imagined movement, left fist vs right fist
    [5, 9, 13],               # 2: real movement,     both fists vs both feet
    [6, 10, 14],              # 3: imagined movement, both fists vs both feet
    [3, 7, 11, 4, 8, 12],     # 4: real + imagined,   left fist vs right fist
    [5, 9, 13, 6, 10, 14],    # 5: real + imagined,   both fists vs both feet
]

mne.set_log_level("ERROR")  # MNE is very talkative; only show real problems


def find_experiment(run):
    """Return the runs of the first experiment that contains `run`."""
    for runs in EXPERIMENTS:
        if run in runs:
            return runs
    raise ValueError(f"run {run} is not a motor task (use a run between 3 and 14)")


def load_raw(subject, runs):
    """Download (first time only) and open the runs of one subject as one Raw."""
    files = eegbci.load_data(subject, runs, path=DATA_PATH, update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)  # "Fc5." -> "FC5"
    raw.set_montage("standard_1005")
    return raw


def filter_raw(raw):
    """Keep only the 7-30 Hz band. Returns a new Raw (the original is untouched)."""
    return raw.copy().filter(LOW_FREQ, HIGH_FREQ, picks="eeg")


def get_epochs(subject, runs):
    """Full preprocessing: load -> filter -> epochs. Returns X (epochs, channels, times) and y."""
    unfiltered_raw = load_raw(subject, runs)
    # print("Unfitered raw : \n", unfiltered_raw)
    filtered_raw = filter_raw(unfiltered_raw)
    # print("Fitered raw : \n", filtered_raw)
    # raw = filter_raw(load_raw(subject, runs))
    # T0 = dropped. T1/T2 -> labels 1 and 2.
    events, _ = mne.events_from_annotations(filtered_raw, event_id={"T1": 1, "T2": 2})
    # print("This are events :\n", events)
    epochs = mne.Epochs(filtered_raw, events, tmin=TMIN, tmax=TMAX, picks="eeg",
                        baseline=None, preload=True)
    # print("This are epochs :\n", epochs)
    X = epochs.get_data(copy=False)
    # print("This is X :\n", X[0][3])
    y = epochs.events[:, -1]
    # print("This is y :\n", y[0])
    return X, y


def mean_class_covariance(X, y, label):
    """Average the epoch covariance matrices belonging to one task label."""
    class_epochs = X[y == label]
    if len(class_epochs) == 0:
        raise ValueError(f"No epochs found for label {label}")
    covariances = [np.cov(epoch) for epoch in class_epochs]
    return np.mean(covariances, axis=0)


def main():
    # Import the plotting helper only when running this demonstration.
    from visualize import print_heat_map, plot_class_covariances

    X, y = get_epochs(1, [4, 8, 12])
    print("X shape (epochs, channels, times):", X.shape)
    epoch = X[0][0]
    # print("Epoch :\n", epoch)
    print("Epoch length is : ", len(epoch))
    # print_heat_map(epoch)
    print("label : ", y[0])
    assert X.ndim == 3 and X.shape[0] == len(y) and set(y.tolist()) == {1, 2}

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
