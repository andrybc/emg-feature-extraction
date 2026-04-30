"""Plotting helpers for EMG learning notebooks."""

import matplotlib.pyplot as plt


def plot_emg_channel(time, emg, channel=0, title="Raw EMG-like signal"):
    """Plot one EMG channel over time."""
    plt.figure(figsize=(12, 4))
    plt.plot(time, emg[:, channel], linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (simulated volts)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_all_channels(time, emg, title="Multi-channel EMG-like signal"):
    """Plot all channels stacked with offsets so they are easier to see."""
    plt.figure(figsize=(12, 6))
    for ch in range(emg.shape[1]):
        offset = ch * 0.8
        plt.plot(time, emg[:, ch] + offset, linewidth=0.7, label=f"Ch {ch + 1}")
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude plus visual offset")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=2)
    plt.show()
