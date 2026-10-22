"""Illustrate signal frequency versus sampling frequency on ONE channel.

Run:   python sampling_demo.py
Save:  python sampling_demo.py --save sampling_demo.png

This uses an invented sine wave, not a recording from the EEG dataset.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, Slider


SAMPLING_RATE = 160  # Measurements per second; this stays fixed.
DURATION = 1.0       # Seconds shown before zooming.
AMPLITUDE = 20.0     # Microvolts, chosen for this illustration.


def make_figure():
    # Exactly 160 measurement times: 0, 1/160, ..., 159/160 seconds.
    sample_times = np.arange(int(SAMPLING_RATE * DURATION)) / SAMPLING_RATE

    # A dense curve helps us visualize the underlying example wave.
    # These extra points are for drawing only, not recorded measurements.
    curve_times = np.linspace(0, DURATION, 4001)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.78, bottom=0.40)
    fig.suptitle("One channel: 160 measurements in one second")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (microvolts)")
    ax.set_xlim(0, DURATION)
    ax.set_ylim(-25, 25)  # Fixed scale makes frequency changes easy to compare.
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.grid(alpha=0.2)

    # Both curves start empty; update() supplies their values below.
    wave_line, = ax.plot([], [], color="tab:blue", label="Underlying example wave")
    sample_dots, = ax.plot(
        [], [], "o", color="tab:orange", markersize=4,
        label="Recorded measurements (160 per second)",
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02))

    slider_ax = fig.add_axes([0.25, 0.20, 0.60, 0.04])
    frequency_slider = Slider(
        slider_ax, "Wave frequency (Hz)", 2, 30, valinit=10, valstep=1,
    )
    checkbox_ax = fig.add_axes([0.10, 0.07, 0.43, 0.08])
    zoom_checkbox = CheckButtons(
        checkbox_ax, ["Zoom to the first 0.1 second"], [False],
    )
    status = fig.text(0.10, 0.28, "")
    fig.text(0.56, 0.10, "Synthetic signal, not real EEG.\nSampling rate stays fixed at 160 Hz.")

    def update(_=None):
        frequency = frequency_slider.val

        # Frequency controls how many times the wave repeats per second.
        curve_values = AMPLITUDE * np.sin(2 * np.pi * frequency * curve_times)
        sample_values = AMPLITUDE * np.sin(2 * np.pi * frequency * sample_times)
        wave_line.set_data(curve_times, curve_values)
        sample_dots.set_data(sample_times, sample_values)

        zoomed = zoom_checkbox.get_status()[0]
        ax.set_xlim(0, 0.1 if zoomed else DURATION)
        status.set_text(
            f"Wave: {frequency:.0f} cycles/second   |   "
            f"Measurement interval: {1 / SAMPLING_RATE:.5f} seconds"
        )
        fig.canvas.draw_idle()

    frequency_slider.on_changed(update)
    zoom_checkbox.on_clicked(update)
    # Keep the controls alive while the plot window is open.
    fig.sampling_controls = (frequency_slider, zoom_checkbox)
    update()
    return fig


def make_mixture_figure():
    """Show several rhythms adding together within one synthetic channel."""
    times = np.arange(int(SAMPLING_RATE * DURATION)) / SAMPLING_RATE
    frequencies = (3, 10, 40)
    amplitudes = (25, 15, 8)  # Microvolts; arbitrary illustration values.
    colors = ("tab:purple", "tab:green", "tab:orange")
    components = [
        amplitude * np.sin(2 * np.pi * frequency * times)
        for frequency, amplitude in zip(frequencies, amplitudes)
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.24, hspace=0.55)
    fig.suptitle("The same channel can contain several frequencies")
    titles = (
        "1. Ingredients: waves contributing to ONE channel",
        "2. Before filtering: add the selected waves at each measurement time",
        "3. After ideal 7-30 Hz filtering: retain only the selected 10 Hz wave",
    )
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("Voltage (µV)")
        ax.set_xlim(0, DURATION)
        ax.set_ylim(-55, 55)  # All panels use the same scale.
        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (seconds)")

    ingredient_lines = []
    for frequency, values, color in zip(frequencies, components, colors):
        line, = axes[0].plot(times, values, color=color, label=f"{frequency} Hz")
        ingredient_lines.append(line)
    axes[0].legend(loc="upper right", ncol=3)
    before_line, = axes[1].plot([], [], color="tab:blue")
    after_line, = axes[2].plot([], [], color="tab:green")

    checkbox_ax = fig.add_axes([0.10, 0.06, 0.27, 0.11])
    components_checkbox = CheckButtons(
        checkbox_ax, ["Include 3 Hz", "Include 10 Hz", "Include 40 Hz"],
        [True, True, True],
    )
    fig.text(
        0.42, 0.08,
        "Each output still has 160 measurements per second.\n"
        "Synthetic example: no mixing between different channels.\n"
        "Ideal removal illustrated here; a real filter has gradual transitions.",
    )

    def update(_=None):
        selected = components_checkbox.get_status()
        before = np.zeros_like(times)
        after = np.zeros_like(times)
        for enabled, frequency, values, line in zip(
            selected, frequencies, components, ingredient_lines,
        ):
            line.set_visible(enabled)
            if enabled:
                before += values
                if 7 <= frequency <= 30:
                    after += values
        before_line.set_data(times, before)
        after_line.set_data(times, after)
        fig.canvas.draw_idle()

    components_checkbox.on_clicked(update)
    fig.mixture_controls = components_checkbox
    update()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PATH", help="Save an image without opening a window")
    args = parser.parse_args()
    fig = make_figure()
    mixture_fig = make_mixture_figure()
    if args.save:
        path = Path(args.save)
        mixture_path = path.with_name(f"{path.stem}_mixture{path.suffix}")
        fig.savefig(path, dpi=200)
        mixture_fig.savefig(mixture_path, dpi=200)
        plt.close(fig)
        plt.close(mixture_fig)
        print(f"Saved {path} and {mixture_path}")
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
