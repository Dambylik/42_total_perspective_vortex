"""Load EEG from PhysioNet, filter it, and cut it into labeled epochs (X, y)."""
import mne
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
    raw = filter_raw(load_raw(subject, runs))
    # T0 = rest, we drop it. T1/T2 = the two movements -> labels 1 and 2.
    events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2})
    epochs = mne.Epochs(raw, events, tmin=TMIN, tmax=TMAX, picks="eeg",
                        baseline=None, preload=True)
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]
    return X, y


if __name__ == "__main__":
    X, y = get_epochs(1, [4, 8, 12])
    print("X shape (epochs, channels, times):", X.shape)
    print("labels:", set(y.tolist()))
    assert X.ndim == 3 and X.shape[0] == len(y) and set(y.tolist()) == {1, 2}
