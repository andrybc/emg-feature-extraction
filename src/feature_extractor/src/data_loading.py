"""
data_loading.py
===============
Load EMG CSV files produced by the EMG simulator app.

The simulator writes CSV files with one row per sample. The exact
columns depend on how you set up the simulator's save dialog, but we
try to handle the common formats automatically:

  Format 1: just channels
      ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8

  Format 2: with a timestamp or index column
      time, ch1, ch2, ..., ch8

  Format 3: with a gesture label column
      ch1, ch2, ..., ch8, gesture

  Format 4: everything
      time, ch1, ch2, ..., ch8, gesture

The loader auto-detects which columns are EMG channels vs metadata
based on standard column name patterns.
"""

import numpy as np
import pandas as pd


# Column name patterns that we recognize as NOT being EMG channels.
# Anything matching these (case-insensitive) is treated as metadata.
METADATA_PATTERNS = [
    "time", "timestamp", "t", "sample", "index", "idx",
    "gesture", "label", "class", "y", "target",
]


def load_emg_csv(csv_path):
    """
    Load an EMG CSV file from the simulator.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    result : dict with keys:
        'signal'    : np.ndarray of shape (n_samples, n_channels)
                      The EMG data ready for windowing.
        'channels'  : list of str
                      Names of the channel columns.
        'gestures'  : np.ndarray of shape (n_samples,) or None
                      Per-sample gesture labels if a label column was
                      found, otherwise None.
        'timestamps': np.ndarray of shape (n_samples,) or None
                      Per-sample timestamps if a time column was found,
                      otherwise None.
        'n_samples' : int
        'n_channels': int
        'fs_estimate': float or None
                      Estimated sampling rate based on timestamps if
                      available, else None.
    """
    # pandas auto-detects the header row and column types.
    df = pd.read_csv(csv_path)

    # Classify each column as channel, gesture, time, or other.
    channel_cols = []
    gesture_col = None
    time_col = None

    for col in df.columns:
        col_lower = col.lower().strip()

        # Check if it matches a metadata pattern.
        is_metadata = False
        for pattern in METADATA_PATTERNS:
            if pattern == col_lower or col_lower.startswith(pattern + "_"):
                is_metadata = True
                if pattern in ("gesture", "label", "class", "y", "target"):
                    gesture_col = col
                elif pattern in ("time", "timestamp", "t"):
                    time_col = col
                break

        # Anything that isn't metadata and contains numbers is a channel.
        if not is_metadata:
            # Sanity check: column should be numeric.
            if pd.api.types.is_numeric_dtype(df[col]):
                channel_cols.append(col)

    if len(channel_cols) == 0:
        raise ValueError(
            f"No EMG channel columns detected in {csv_path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Extract the EMG signal as a NumPy array. Shape: (n_samples, n_channels).
    # Using .values gives us a plain array without the pandas overhead.
    signal = df[channel_cols].values.astype(np.float64)

    # Extract gesture labels if available.
    gestures = df[gesture_col].values if gesture_col else None

    # Extract timestamps and estimate sampling rate if available.
    timestamps = None
    fs_estimate = None
    if time_col:
        timestamps = df[time_col].values.astype(np.float64)
        if len(timestamps) > 1:
            # Estimate fs from the median time delta. Median is more
            # robust to gaps or jitter than mean.
            dt = np.median(np.diff(timestamps))
            if dt > 0:
                # If timestamps are in seconds, fs = 1/dt.
                # If timestamps look like sample indices (integer steps),
                # we can't recover fs from them.
                if dt < 1.0:  # likely seconds, not indices
                    fs_estimate = 1.0 / dt

    return {
        "signal": signal,
        "channels": channel_cols,
        "gestures": gestures,
        "timestamps": timestamps,
        "n_samples": signal.shape[0],
        "n_channels": signal.shape[1],
        "fs_estimate": fs_estimate,
    }


def assign_window_labels(gestures, window_size, step_size, mode="majority"):
    """
    Given a per-sample gesture array, assign one label per window.

    Why this matters: after windowing, you have N windows but the
    original CSV had per-sample labels. For supervised learning you
    need exactly one label per window. There are two common conventions:

      'majority': use the most common gesture in the window.
                  Good for clean static training data.
      'last':     use the gesture at the last sample of the window.
                  Good for real-time control where the latest sample
                  matters most.

    Parameters
    ----------
    gestures : np.ndarray of shape (n_samples,)
        Per-sample gesture labels (strings or ints).
    window_size : int
    step_size : int
    mode : str, 'majority' or 'last'

    Returns
    -------
    window_labels : np.ndarray of shape (n_windows,)
    """
    n_samples = len(gestures)
    n_windows = (n_samples - window_size) // step_size + 1
    labels = np.empty(n_windows, dtype=object)

    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window_labels_slice = gestures[start:end]

        if mode == "last":
            labels[i] = window_labels_slice[-1]
        else:  # majority
            # Find the most common label in this window.
            unique, counts = np.unique(window_labels_slice, return_counts=True)
            labels[i] = unique[np.argmax(counts)]

    return labels
