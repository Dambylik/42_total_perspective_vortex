"""Look at the EEG before and after filtering.

    python visualize.py <subject> <run>      default: subject 1, run 4
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from preprocess import HIGH_FREQ, LOW_FREQ, filter_raw, load_raw


ANNOTATION_COLORS = {"T0": "tab:blue", "T1": "tab:orange", "T2": "tab:green"}
IMAGE_DIR = Path(__file__).resolve().parent / "docs" / "images"


def print_heat_map(epoch):
    fig, ax = plt.subplots(figsize=(10, 8))
    covariance = np.cov(epoch)
    covariance_uv = covariance * 1e12
    limit = np.abs(covariance_uv).max()

    image = ax.imshow(
        covariance_uv,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )

    ax.set_title("Covariance between 64 channels — first epoch")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Channel index")

    fig.colorbar(image, ax=ax, label="Covariance (µV²)")
    fig.tight_layout()

    save_figure(fig, "heat_map.png")
    plt.show()
    return fig


def plot_class_covariances(first_covariance, second_covariance, run, counts):
    """Compare class averages in µV²; the difference uses its own colour scale."""
    labels = get_annotation_labels(run)
    matrices = (first_covariance * 1e12, second_covariance * 1e12)
    limit = max(np.abs(matrix).max() for matrix in matrices) or 1.0
    difference = matrices[0] - matrices[1]
    difference_limit = np.abs(difference).max() or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), layout="constrained")
    fig.suptitle("Average epoch covariances by task — exploratory comparison")
    for ax, matrix, label, count in zip(
        axes[:2], matrices, (labels["T1"], labels["T2"]), counts,
    ):
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"{label}\n{count} epochs — shared scale", fontsize=11)
    fig.colorbar(image, ax=list(axes[:2]), label="Mean covariance (µV²)", shrink=0.7)

    difference_image = axes[2].imshow(
        difference, cmap="RdBu_r", vmin=-difference_limit, vmax=difference_limit,
    )
    axes[2].set_title("T1 average minus T2 average\nSeparate difference scale", fontsize=11)
    fig.colorbar(
        difference_image, ax=axes[2], label="Covariance difference (µV²)", shrink=0.7,
    )
    for ax in axes:
        ax.set_xlabel("Channel index")
        ax.set_ylabel("Channel index")
    save_figure(fig, "class_covariances.png")
    plt.show()
    return fig


def save_figure(fig, filename, **kwargs):
    """Keep generated figures together with the README images."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_DIR / filename, dpi=200, **kwargs)


def get_annotation_labels(run):
    """Translate annotation codes according to the task recorded in this run."""
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
    return {"T0": "Rest", "T1": task_labels[0], "T2": task_labels[1]}


def print_recording_info(raw):
    """Print recording metadata and the first ten task annotations."""
    print(raw.info)
    print("channels:", raw.ch_names)
    for annotation in raw.annotations[:10]:
        print(
            "Start:", annotation["onset"],
            "Duration:", annotation["duration"],
            "Code:", annotation["description"],
        )


def save_sensor_plot(raw):
    """Draw and save the electrode locations."""
    fig = raw.plot_sensors(show_names=True, title="64 electrodes (10-10 system)")
    save_figure(fig, "sensor_positions.png")
    return fig


def save_signal_plot(recording, title, filename, subject, run):
    """Save time signals with a fixed voltage scale and a task legend."""
    annotation_labels = get_annotation_labels(run)
    fig = recording.plot(
        title=title, scalings={"eeg": 100e-6}, show=False,
        annotation_colors=ANNOTATION_COLORS,
    )
    handles = [
        Patch(facecolor=color, alpha=0.4, label=f"{code}: {annotation_labels[code]}")
        for code, color in ANNOTATION_COLORS.items()
        if code in recording.annotations.description
    ]
    legend = fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        title=f"Subject {subject}, run {run} - recorded task annotations",
    )
    fig.canvas.draw()
    legend_bottom = legend.get_window_extent(fig.canvas.get_renderer()).y0
    fig.subplots_adjust(top=legend_bottom / fig.bbox.height - 0.02)
    fig.canvas.draw()
    save_figure(fig, filename, bbox_inches="tight")
    return fig


def save_spectrum_plot(recording, title, filename):
    """Draw and save power density across frequencies for all channels."""
    fig = recording.compute_psd().plot()
    fig.suptitle(title)
    save_figure(fig, filename)
    return fig


def print_channel_power(raw, channel="C3", fmin=7, fmax=30):
    """Print original power density and its peak within the requested band."""
    spectrum = raw.compute_psd(picks=[channel], fmin=fmin, fmax=fmax)
    power, frequencies = spectrum.get_data(return_freqs=True)
    power = power[0] * 1_000_000_000_000  # V²/Hz -> µV²/Hz

    for frequency, value in zip(frequencies, power):
        print(f"{frequency:.2f} Hz: {value:.4f} µV²/Hz")
    peak_index = power.argmax()
    print("Strongest frequency:", frequencies[peak_index], "Hz")
    print("Power density:", power[peak_index], "µV²/Hz")


def visualize_recording(raw, subject, run):
    """Inspect a loaded recording and save the five before/after figures."""
    print_recording_info(raw)
    save_sensor_plot(raw)
    save_signal_plot(raw, "raw signal", "signal_before.png", subject, run)
    save_spectrum_plot(raw, "power spectrum BEFORE filtering", "spectrum_before.png")
    # print_channel_power(raw)

    filtered = filter_raw(raw)
    save_signal_plot(
        filtered, f"filtered {LOW_FREQ}-{HIGH_FREQ} Hz", "signal_after.png",
        subject, run,
    )
    save_spectrum_plot(
        filtered, f"power spectrum AFTER {LOW_FREQ}-{HIGH_FREQ} Hz filter",
        "spectrum_after.png",
    )


def main():
    """Read command-line arguments, load the data and display its plots."""
    subject = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    raw = load_raw(subject, [run])
    visualize_recording(raw, subject, run)
    # plt.show()


if __name__ == "__main__":
    main()
