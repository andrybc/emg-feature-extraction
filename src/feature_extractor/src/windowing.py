"""
windowing.py
============
Split a continuous EMG signal into short overlapping windows.

Why we window:
  A raw recording is a long time-series (thousands of samples). We
  can't hand 80,000 numbers to a classifier and expect a useful
  prediction, and we can't hand 1 number either (too little info).
  The middle ground: slice into short windows (typically 150 to 250 ms),
  compute a small set of statistics per window, and classify the
  statistics.

Why short windows specifically:
  - Long enough for stable feature estimates (Hudgins showed 200 ms
    gives a good signal-to-noise ratio for the statistics).
  - Short enough that total system latency (window + feature compute
    + classifier) stays under 300 ms, which is the threshold above
    which users start perceiving lag in a prosthesis.

Why overlap (step < window):
  - Overlapping windows give MORE training examples per second of data.
  - Smoother prediction stream during real-time use because predictions
    update every step_size, not every window_size.
  - Typical: 200 ms window, 50 ms step => 4x overlap.
"""

import numpy as np


def window_signal(signal, window_size, step_size):
    """
    Slice a continuous signal into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray of shape (n_samples, n_channels)
        The raw EMG signal. Rows are time, columns are channels.
    window_size : int
        Number of samples per window. Example: at 1000 Hz, 200 samples = 200 ms.
    step_size : int
        Number of samples between window starts. step_size < window_size
        gives overlapping windows. Example: at 1000 Hz, 50 samples = 50 ms.

    Returns
    -------
    windows : np.ndarray of shape (n_windows, window_size, n_channels)
        A 3D array where the first axis indexes the window, the second
        axis is time within the window, and the third axis is channel.
    """
    # Handle 1D input by adding a channel axis (so the code below is uniform).
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    n_samples = signal.shape[0]
    n_channels = signal.shape[1]

    # How many full windows fit? The first window starts at 0, the last
    # window starts at (n_samples - window_size). Between those, we step
    # by step_size. So the count is floor((N - W) / S) + 1.
    n_windows = (n_samples - window_size) // step_size + 1

    if n_windows <= 0:
        raise ValueError(
            f"Signal too short: {n_samples} samples is less than "
            f"window_size={window_size}. Increase signal length or "
            f"decrease window size."
        )

    # Pre-allocate the output array. Allocating once and filling in is
    # much faster than appending row by row in a loop.
    windows = np.empty(
        (n_windows, window_size, n_channels),
        dtype=signal.dtype
    )

    # Fill each window by slicing.
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = signal[start:end]

    return windows
