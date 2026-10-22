"""Look at the EEG before and after filtering (required by the subject).

    python visualize.py [subject] [run]      default: subject 1, run 4
"""
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from preprocess import HIGH_FREQ, LOW_FREQ, filter_raw, load_raw

subject = int(sys.argv[1]) if len(sys.argv) > 1 else 1
run = int(sys.argv[2]) if len(sys.argv) > 2 else 4

# The annotation meaning depends on the task recorded in this run.
annotation_colors = {"T0": "tab:blue", "T1": "tab:orange", "T2": "tab:green"}
if run in (3, 7, 11):
    task_labels = ("Actual left-fist movement", "Actual right-fist movement")
elif run in (4, 8, 12):
    task_labels = ("Imagined left-fist movement", "Imagined right-fist movement")
elif run in (5, 9, 13):
    task_labels = ("Actual both-fists movement", "Actual both-feet movement")
elif run in (6, 10, 14):
    task_labels = ("Imagined both-fists movement", "Imagined both-feet movement")
else:
    task_labels = ("Not used in baseline run", "Not used in baseline run")

annotation_labels = {"T0": "Rest", "T1": task_labels[0], "T2": task_labels[1]}


def save_signal_plot(recording, title, filename):
    fig = recording.plot(
        title=title, scalings={"eeg": 100e-6}, show=False,
        annotation_colors=annotation_colors,
    )
    handles = [
        Patch(facecolor=color, alpha=0.4, label=f"{code}: {annotation_labels[code]}")
        for code, color in annotation_colors.items()
        if code in recording.annotations.description
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
        title=f"Subject {subject}, run {run} - recorded task annotations",
    )
    # Complete MNE's layout before exporting channel names and axes.
    fig.canvas.draw()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    return fig


raw = load_raw(subject, [run])
print(raw.info)
print("channels:", raw.ch_names)
for annotation in raw.annotations[:10]:
    print(
        "Start:", annotation["onset"],
        "Duration:", annotation["duration"],
        "Code:", annotation["description"],
    )

fig1 = raw.plot_sensors(show_names=True, title="64 electrodes (10-10 system)")
fig1.savefig("sensor_positions.png", dpi=200)


# BEFORE: slow drift + 60 Hz mains noise are visible in the spectrum
fig_before = save_signal_plot(raw, "raw signal", "signal_before.png")
fig2 = raw.compute_psd().plot()

plt.suptitle("power spectrum BEFORE filtering")
fig2.savefig("spectrum_before.png", dpi=200)
spectrum = raw.compute_psd(
    picks=["C3"],
    fmin=7,
    fmax=30,
)

power, frequencies = spectrum.get_data(return_freqs=True)

power = power[0] * 1_000_000_000_000

for frequency, value in zip(frequencies, power):
    print(f"{frequency:.2f} Hz: {value:.4f} µV²/Hz")
peak_index = power.argmax()

print("Strongest frequency:", frequencies[peak_index], "Hz")
print("Power density:", power[peak_index], "µV²/Hz")

# AFTER: only the 7-30 Hz band (mu + beta rhythms) is left
filtered = filter_raw(raw)
fig_after = save_signal_plot(
    filtered, f"filtered {LOW_FREQ}-{HIGH_FREQ} Hz", "signal_after.png",
)
fig3 = filtered.compute_psd().plot()
plt.suptitle(f"power spectrum AFTER {LOW_FREQ}-{HIGH_FREQ} Hz filter")
fig3.savefig("spectrum_after.png", dpi=200)

plt.show()  # blocks so the windows stay open
