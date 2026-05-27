"""
pipeline.py
===========
Orchestrates the full extraction: raw signal -> windows -> feature matrix.

This file controls WHICH features are computed. Both Path A (Hudgins)
and Path B (Phinyomark) are defined here. Path A is ACTIVE by default;
Path B is COMMENTED OUT until you're ready to compare.

How to enable Path B:
  1. Scroll to the function `extract_features_path_b()` below.
  2. Uncomment the lines marked with "# UNCOMMENT FOR PATH B".
  3. In the GUI app, the "Path B" and "Both" radio buttons will then
     produce real output instead of raising NotImplementedError.

Output format:
  The feature matrix has shape (n_windows, n_features). Each row is
  one observation that can go straight into sklearn. The column
  ordering convention is:
      [ch0_feat0, ch0_feat1, ..., ch0_featK,
       ch1_feat0, ch1_feat1, ..., ch1_featK,
       ...]
  i.e., features are grouped by channel. This convention is consistent
  with what we'll use in Parts 4 through 10 of the pipeline.
"""

import numpy as np

# Import all feature functions. Even ones we aren't using yet are
# imported so they're ready when you flip the Path B switch.
from .features import (
    mav,
    waveform_length,
    zero_crossings,
    slope_sign_changes,
    willison_amplitude,   # Path B only
    ar_coefficients,      # Path B only
)
from .windowing import window_signal


# ---------------------------------------------------------------------
# PATH A: Hudgins (1993) - ACTIVE BY DEFAULT
# ---------------------------------------------------------------------

def extract_features_path_a(signal, window_size, step_size,
                             zc_threshold=1e-5,
                             ssc_threshold=1e-5,
                             channel_names=None):
    """
    Extract the Hudgins four-feature set (MAV, WL, ZC, SSC) per channel.

    For an 8-channel signal this gives 4 * 8 = 32 features per window.

    Parameters
    ----------
    signal : np.ndarray of shape (n_samples, n_channels)
    window_size : int, samples per window
    step_size : int, samples between window starts
    zc_threshold : float, noise threshold for zero crossings
    ssc_threshold : float, noise threshold for slope sign changes
    channel_names : list of str, optional
        Original channel column names (e.g., ['ch1_mV', 'ch2_mV', ...]).
        If provided, output feature columns are named like 'ch1_mV_mav'
        so they unambiguously trace back to the input. If None, falls
        back to generic 'ch0_mav', 'ch1_mav', etc.

    Returns
    -------
    feature_matrix : np.ndarray of shape (n_windows, 4 * n_channels)
    feature_names : list of str, names for each column
    """
    # Step 1: chop the signal into overlapping windows.
    # Shape becomes (n_windows, window_size, n_channels).
    windows = window_signal(signal, window_size, step_size)
    n_windows = windows.shape[0]
    n_channels = windows.shape[2]

    # Step 2: pre-allocate the feature matrix.
    # 4 features per channel.
    n_features_per_channel = 4
    feature_matrix = np.empty((n_windows, n_features_per_channel * n_channels))

    # Step 3: loop over windows. For each window, compute each feature
    # for all channels at once (the feature functions handle multi-channel
    # input natively via axis=0).
    for w_idx in range(n_windows):
        window = windows[w_idx]  # shape (window_size, n_channels)

        # Compute each feature. Each returns an array of shape (n_channels,)
        # because the feature functions reduce along axis 0 (time).
        f_mav = mav(window)
        f_wl = waveform_length(window)
        f_zc = zero_crossings(window, threshold=zc_threshold)
        f_ssc = slope_sign_changes(window, threshold=ssc_threshold)

        # Stack the features so we have shape (n_channels, n_features_per_channel).
        # axis=1 means "make each feature a column". Row k of the stack
        # is "all 4 features for channel k".
        stacked = np.stack([f_mav, f_wl, f_zc, f_ssc], axis=1)

        # Flatten in row-major order so the layout matches the convention:
        # [ch0_f0, ch0_f1, ch0_f2, ch0_f3, ch1_f0, ch1_f1, ...]
        feature_matrix[w_idx] = stacked.flatten()

    # Step 4: build human-readable column names for the output CSV.
    # If channel_names were provided by the caller (e.g., from the CSV
    # loader), use them directly so feature columns trace back to the
    # input columns unambiguously. Otherwise fall back to generic names.
    feature_names = []
    for c in range(n_channels):
        # Pick a stem: either the original column name (e.g., 'ch1_mV')
        # or a generic fallback ('ch0', 'ch1', ...).
        stem = channel_names[c] if channel_names is not None else f"ch{c}"
        feature_names.extend([
            f"{stem}_mav",
            f"{stem}_wl",
            f"{stem}_zc",
            f"{stem}_ssc",
        ])

    return feature_matrix, feature_names


# ---------------------------------------------------------------------
# PATH B: Phinyomark (2012) - DISABLED BY DEFAULT
# ---------------------------------------------------------------------
# This function is intentionally a stub. To enable Path B:
#   1. Comment out the line that says `raise NotImplementedError(...)`.
#   2. Uncomment ALL the lines marked "# UNCOMMENT FOR PATH B".
#   3. Save and re-run the app. The Path B and Both options will now work.

def extract_features_path_b(signal, window_size, step_size,
                             wamp_threshold=1e-5,
                             ar_order=4,
                             channel_names=None):
    """
    Extract the Phinyomark non-redundant set (MAV, WL, WAMP, AR-4).

    For an 8-channel signal this gives (3 + 4) * 8 = 56 features per window:
    MAV, WL, WAMP plus 4 AR coefficients, per channel.

    Parameters
    ----------
    signal : np.ndarray of shape (n_samples, n_channels)
    window_size : int
    step_size : int
    wamp_threshold : float, noise threshold for Willison amplitude
    ar_order : int, AR model order (Phinyomark recommends 4)
    channel_names : list of str, optional
        Original channel column names. Same role as in Path A.

    Returns
    -------
    feature_matrix : np.ndarray of shape (n_windows, (3 + ar_order) * n_channels)
    feature_names : list of str
    """

    # ----- ENABLE PATH B BY DELETING THE NEXT LINE -----
    raise NotImplementedError(
        "Path B is disabled. To enable it: open src/pipeline.py, delete "
        "this raise statement, and uncomment the lines marked "
        "'# UNCOMMENT FOR PATH B' below."
    )

    # # UNCOMMENT FOR PATH B
    # # Step 1: chop into windows (same as Path A).
    # windows = window_signal(signal, window_size, step_size)
    # n_windows = windows.shape[0]
    # n_channels = windows.shape[2]
    #
    # # UNCOMMENT FOR PATH B
    # # Step 2: pre-allocate. 3 base features + ar_order AR coefficients per channel.
    # n_features_per_channel = 3 + ar_order
    # feature_matrix = np.empty((n_windows, n_features_per_channel * n_channels))
    #
    # # UNCOMMENT FOR PATH B
    # # Step 3: loop and compute.
    # for w_idx in range(n_windows):
    #     window = windows[w_idx]
    #
    #     # Base features: shape (n_channels,) each.
    #     f_mav = mav(window)
    #     f_wl = waveform_length(window)
    #     f_wamp = willison_amplitude(window, threshold=wamp_threshold)
    #
    #     # AR coefficients: shape (ar_order, n_channels).
    #     # Each column is one channel's coefficient vector.
    #     f_ar = ar_coefficients(window, order=ar_order)
    #
    #     # Stack everything into shape (n_channels, n_features_per_channel).
    #     # We transpose f_ar so each ROW becomes one channel's AR coefficients.
    #     base_stacked = np.stack([f_mav, f_wl, f_wamp], axis=1)  # (n_channels, 3)
    #     ar_stacked = f_ar.T                                       # (n_channels, ar_order)
    #     full_stacked = np.concatenate([base_stacked, ar_stacked], axis=1)
    #
    #     feature_matrix[w_idx] = full_stacked.flatten()
    #
    # # UNCOMMENT FOR PATH B
    # # Step 4: column names. Use original channel names if provided
    # # (same logic as Path A).
    # feature_names = []
    # for c in range(n_channels):
    #     stem = channel_names[c] if channel_names is not None else f"ch{c}"
    #     feature_names.append(f"{stem}_mav")
    #     feature_names.append(f"{stem}_wl")
    #     feature_names.append(f"{stem}_wamp")
    #     for p in range(1, ar_order + 1):
    #         feature_names.append(f"{stem}_ar{p}")
    #
    # # UNCOMMENT FOR PATH B
    # return feature_matrix, feature_names


# ---------------------------------------------------------------------
# COMBINED: Run both paths and concatenate their feature matrices.
# ---------------------------------------------------------------------
# This depends on Path B being enabled. Until then it will raise the
# same NotImplementedError that Path B does.

def extract_features_both(signal, window_size, step_size,
                           zc_threshold=1e-5,
                           ssc_threshold=1e-5,
                           wamp_threshold=1e-5,
                           ar_order=4,
                           channel_names=None):
    """
    Run both paths and concatenate their feature matrices column-wise.

    The Path A columns come first, then the Path B columns. This is
    useful for comparison studies where you want to feed the union
    set to a classifier and let feature selection (or the classifier
    itself) decide what matters.

    Note: MAV and WL appear in BOTH paths, so they will be duplicated
    columns in the output. We keep them duplicated rather than dedup
    because it makes the column ordering predictable and reversible.
    """
    fm_a, names_a = extract_features_path_a(
        signal, window_size, step_size,
        zc_threshold=zc_threshold,
        ssc_threshold=ssc_threshold,
        channel_names=channel_names,
    )
    fm_b, names_b = extract_features_path_b(
        signal, window_size, step_size,
        wamp_threshold=wamp_threshold,
        ar_order=ar_order,
        channel_names=channel_names,
    )

    # Prefix names so it's clear which path each column came from.
    names_a_prefixed = [f"A_{n}" for n in names_a]
    names_b_prefixed = [f"B_{n}" for n in names_b]

    feature_matrix = np.concatenate([fm_a, fm_b], axis=1)
    feature_names = names_a_prefixed + names_b_prefixed

    return feature_matrix, feature_names