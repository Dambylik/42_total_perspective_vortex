"""Look at the EEG before and after filtering (required by the subject).

    python visualize.py [subject] [run]      default: subject 1, run 4
"""
import sys

import matplotlib.pyplot as plt

from preprocess import HIGH_FREQ, LOW_FREQ, filter_raw, load_raw

subject = int(sys.argv[1]) if len(sys.argv) > 1 else 1
run = int(sys.argv[2]) if len(sys.argv) > 2 else 4

raw = load_raw(subject, [run])
print(raw.info)
print("channels:", raw.ch_names)

raw.plot_sensors(show_names=True, title="64 electrodes (10-10 system)")

# BEFORE: slow drift + 60 Hz mains noise are visible in the spectrum
raw.plot(title="raw signal", scalings="auto")
raw.compute_psd().plot()
plt.suptitle("power spectrum BEFORE filtering")

# AFTER: only the 7-30 Hz band (mu + beta rhythms) is left
filtered = filter_raw(raw)
filtered.plot(title=f"filtered {LOW_FREQ}-{HIGH_FREQ} Hz", scalings="auto")
filtered.compute_psd().plot()
plt.suptitle(f"power spectrum AFTER {LOW_FREQ}-{HIGH_FREQ} Hz filter")

plt.show()  # blocks so the windows stay open
