"""
load_emg_recording.py
=====================
Quick utility to load a CSV recording from the EMG Simulator
and verify it is compatible with the notebook preprocessing pipeline.

Usage:
    python load_emg_recording.py path/to/emg_recording.csv

Or import the function in your notebook:
    from load_emg_recording import load_recording
    emg, labels, fs, gesture_names = load_recording('emg_hand_closed_20240101.csv')
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_recording(csv_path: str):
    """
    Load an EMG Simulator CSV recording into numpy arrays.

    Parameters
    ----------
    csv_path : str -- path to the .csv file from the EMG Simulator

    Returns
    -------
    emg          : np.ndarray, shape (n_samples, 8)  -- raw EMG in mV
    labels       : np.ndarray, shape (n_samples,)    -- gesture integer labels
    gesture_names: np.ndarray, shape (n_samples,)    -- gesture name strings
    timestamps   : np.ndarray, shape (n_samples,)    -- time in seconds
    fs           : int                               -- sampling rate (1000 Hz)
    """
    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"  Shape:    {df.shape}  ({df.shape[0]} samples, {df.shape[1]} columns)")
    print(f"  Columns:  {list(df.columns)}")
    print(f"  Duration: {df['timestamp_s'].iloc[-1]:.2f} seconds")
    print(f"  Gestures: {df['gesture_name'].unique()}")

    # Extract EMG channel columns (ch1_mV through ch8_mV)
    ch_cols = [c for c in df.columns if c.startswith('ch') and c.endswith('_mV')]
    emg     = df[ch_cols].values.astype(np.float64)   # (n_samples, 8)

    labels        = df['gesture_label'].values.astype(int)
    gesture_names = df['gesture_name'].values
    timestamps    = df['timestamp_s'].values.astype(np.float64)

    fs = 1000   # Sampling rate (fixed in simulator)

    print(f"\n  EMG array shape:  {emg.shape}  -- (samples, channels)")
    print(f"  Labels shape:     {labels.shape}")
    print(f"  Unique labels:    {np.unique(labels)}")
    print(f"  EMG amplitude:    min={emg.min():.5f} mV, max={emg.max():.5f} mV")
    print(f"\nReady to pass to preprocessing pipeline.")

    return emg, labels, gesture_names, timestamps, fs


def quick_plot(emg, timestamps, gesture_names, n_channels=8):
    """
    Plot the loaded EMG recording (all channels, color-coded by gesture).
    """
    CHANNEL_COLORS = [
        '#34d399', '#38bdf8', '#fb923c', '#f472b6',
        '#a78bfa', '#fbbf24', '#4ade80', '#f87171',
    ]
    GESTURE_COLORS = {
        'Rest':            '#94a3b8',
        'Hand Open':       '#34d399',
        'Hand Closed':     '#f87171',
        'Wrist Flexion':   '#60a5fa',
        'Wrist Extension': '#fbbf24',
    }

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 9),
                              sharex=True, facecolor='#07101f')
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.06, hspace=0.06)

    # Shade gesture regions
    unique_gestures = []
    start_idx = 0
    current   = gesture_names[0]

    for i in range(1, len(gesture_names)):
        if gesture_names[i] != current or i == len(gesture_names) - 1:
            for ax in axes:
                ax.axvspan(
                    timestamps[start_idx], timestamps[i],
                    alpha=0.08,
                    color=GESTURE_COLORS.get(current, '#ffffff'),
                    zorder=0
                )
            unique_gestures.append((current, timestamps[start_idx]))
            start_idx = i
            current   = gesture_names[i]

    for ch in range(n_channels):
        ax = axes[ch]
        ax.set_facecolor('#091322' if ch % 2 == 0 else '#07101f')
        ax.plot(timestamps, emg[:, ch],
                color=CHANNEL_COLORS[ch], linewidth=0.6, alpha=0.9)
        ax.axhline(0, color='#0f2040', linewidth=0.4, linestyle='--')
        ax.set_ylabel(f'CH{ch+1}', color=CHANNEL_COLORS[ch],
                      fontsize=8, fontweight='bold',
                      fontfamily='monospace', rotation=0, labelpad=24)
        ax.yaxis.set_label_coords(-0.042, 0.35)
        for spine in ax.spines.values():
            spine.set_color('#0d1e3a')
        ax.tick_params(colors='#334155', labelsize=6)

    axes[-1].set_xlabel('Time (seconds)', color='#475569', fontsize=9)

    # Build legend from gestures found
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=GESTURE_COLORS.get(g, '#fff'), alpha=0.5, label=g)
        for g in dict.fromkeys(gesture_names)   # preserve order, deduplicate
    ]
    axes[0].legend(handles=legend_patches, loc='upper right', fontsize=8,
                   facecolor='#0f172a', edgecolor='#1e3a5f', labelcolor='#94a3b8')

    fig.suptitle(
        f"EMG Recording -- {n_channels} Channels\n"
        f"{len(emg):,} samples · {timestamps[-1]:.1f}s · 1000 Hz",
        color='#e2e8f0', fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# Run directly: python load_emg_recording.py your_file.csv
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # No file given: look for the most recent CSV in data/simulated/
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        sim_dir     = os.path.join(script_dir, 'data', 'simulated')

        if not os.path.isdir(sim_dir):
            print("Usage: python load_emg_recording.py path/to/recording.csv")
            print("  No data/simulated/ directory found. Record something first.")
            sys.exit(1)

        csvs = sorted(
            [f for f in os.listdir(sim_dir) if f.endswith('.csv')],
            reverse=True   # Most recent first
        )

        if not csvs:
            print(f"No CSV files found in {sim_dir}")
            print("Run the simulator and record some data first.")
            sys.exit(1)

        csv_path = os.path.join(sim_dir, csvs[0])
        print(f"No file specified. Using most recent: {csvs[0]}\n")
    else:
        csv_path = sys.argv[1]

    emg, labels, gesture_names, timestamps, fs = load_recording(csv_path)
    quick_plot(emg, timestamps, gesture_names)