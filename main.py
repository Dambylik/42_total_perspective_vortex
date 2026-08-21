from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

# https://mne.tools/stable/documentation/datasets.html#datasets
# EDF = European Data Format, the standard file format for EEG recordings.
# Load_data (get files) → read_raw_edf (one run → Raw) → concatenate_raws (join runs) → standardize (fix names).

subjects = [1]  # # which person (1–109 available)
runs = [4, 8, 12]  # which recordings for that person
raw_fnames = eegbci.load_data(subjects, runs, path="./data", update_path=False) # downloads the .edf files from PhysioNet (only the first time) and returns their file paths — it does not load the signal itself.
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
# opens one .edf file into a Raw object — MNE's in-memory container for a continuous recording (the voltage matrix + metadata: channel names, sample rate, event markers). preload=True means "read the actual samples into RAM now" (instead of lazily). One read_raw_edf call = one run.
raw = concatenate_raws(raws)
# glue the 3 runs into one continuous Raw
eegbci.standardize(raw) # rename channels (e.g. "Fc5." -> "FC5") to standard names

print('-' * 80)
print("RAW INFO :" , raw.info)
print('-' * 80) 
print("64 electrode names: \n",raw.ch_names)
print('-' * 80)
from mne import events_from_annotations 
events, event_id = events_from_annotations(raw)
print("T0 = rest (no movement),\nT1 = movement #1 (in runs 4/8/12: left fist),\nT2 = movement #2 (in runs 4/8/12: right fist)\n: \n", event_id)
print("events shape:", events.shape)   # e.g. (90, 3)
print("first 10 events:\n, col 0: SAMPLE NUMBER when it happened\n, col 1: previous event's code (0 = none)\n, col 3: Event CODE  (1=T0, 2=T1, 3=T2)  ← the label\n", events[:10])
#The only column you feed a classifier is col 2 (the label);
# col 0 tells MNE where to slice when you make epochs next.
print('-' * 80)

# --- SEE the data before filtering ---
from mne.channels import make_standard_montage
raw.set_montage(make_standard_montage("standard_1005"))  # attach real 3-D electrode positions

raw.plot()                        # continuous signal, scrollable (drag/scroll to inspect)
raw.plot_sensors(show_names=True) # montage: where the 64 electrodes sit on the scalp
raw.compute_psd().plot()          # power spectral density: power per frequency

import matplotlib.pyplot as plt
plt.show()                        # ponytail: blocks so the windows stay open



def main():
    pass
 
if __name__ == "__main__":
    main()
    