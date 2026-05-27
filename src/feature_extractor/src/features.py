"""
features.py
============
Time-domain EMG feature extraction functions.

This module contains the building blocks for both feature paths:

  Path A (Hudgins 1993, IEEE TBME 40(1)):
      MAV, WL, ZC, SSC
      The classical four-feature set used as the baseline in nearly
      every myoelectric pattern-recognition paper since 1993.

  Path B (Phinyomark 2012, ESWA 39):
      MAV, WL, WAMP, AR(4)
      The "non-redundant" recommendation. Phinyomark showed that VAR
      duplicates MAV (both energy), and SSC duplicates ZC (both
      frequency), so a smaller but better-chosen set performs as well
      or better than Hudgins.

By default ONLY Path A is active. WAMP and AR are defined here but
they are disabled in the feature pipeline (see pipeline.py). To turn
on Path B, follow the instructions at the bottom of pipeline.py.

Each feature function accepts either:
  - a 1D window of shape (W,) for a single channel, or
  - a 2D window of shape (W, C) for multi-channel data
and returns either a scalar (1D input) or an array of shape (C,)
(2D input). This is what lets us use the same function on one channel
during testing and on all 8 channels in production.
"""

import numpy as np

# SciPy is only needed for the AR feature (Path B). Importing at the top
# is fine because it does not run any AR code unless ar_coefficients()
# is actually called.
from scipy.linalg import solve_toeplitz


# ---------------------------------------------------------------------
# PATH A: Hudgins (1993) feature set
# ---------------------------------------------------------------------
# These four features form the canonical baseline. Cheap to compute,
# easy to interpret, well documented in the literature.

def mav(window):
    """
    Mean Absolute Value (Hudgins Eq. 1).

    Rectify the signal (take absolute value of every sample) then
    average. This is a proxy for muscle contraction intensity: when
    the muscle contracts harder, the EMG signal swings further from
    zero, so the average of |x| is larger.

    Math:  MAV = (1/N) * sum(|x_i|)
    """
    return np.mean(np.abs(window), axis=0)


def waveform_length(window):
    """
    Waveform Length (Hudgins Eq. 5).

    Sum of the absolute differences between consecutive samples.
    Hudgins called this "a measure of waveform amplitude, frequency,
    and duration all within a single parameter" because a high-
    amplitude OR high-frequency signal traces a longer path.

    Math:  WL = sum(|x_{i+1} - x_i|)
    """
    # np.diff computes x[1:] - x[:-1] in one shot. axis=0 means
    # "do this along the time axis" so it works on multi-channel
    # windows too.
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def zero_crossings(window, threshold=1e-5):
    """
    Zero Crossings with noise threshold (Hudgins Eq. 3).

    Count how many times the signal crosses the zero line, but ONLY
    count crossings where the amplitude change is big enough to rule
    out tiny noise wiggles. The threshold is the "noise dead zone."

    Intuition: more zero crossings means higher dominant frequency.

    A crossing occurs between samples x[i] and x[i+1] when:
      (1) their signs differ (so the line was crossed), AND
      (2) |x[i] - x[i+1]| >= threshold (to ignore noise wiggles)
    """
    x = window

    # Trick: if two consecutive samples have OPPOSITE signs, their
    # product is negative. This vectorizes the sign-change check
    # across the whole window in one operation instead of a loop.
    sign_change = (x[:-1] * x[1:]) < 0

    # Amplitude difference must exceed the noise floor.
    amp_change = np.abs(x[:-1] - x[1:]) >= threshold

    # Both conditions must hold. np.sum on a boolean array counts True.
    return np.sum(sign_change & amp_change, axis=0)


def slope_sign_changes(window, threshold=1e-5):
    """
    Slope Sign Changes with noise threshold (Hudgins Eq. 4).

    Count how many times the slope of the signal reverses direction
    (i.e., the signal hits a local max or local min). Complements ZC:
    where ZC counts events at the zero line, SSC counts peaks and
    valleys ANYWHERE on the signal.

    A slope sign change occurs at sample x[k] when:
      (1) x[k] is either higher than both neighbors (local max) or
          lower than both neighbors (local min), AND
      (2) at least one of the neighbor differences exceeds threshold.
    """
    x = window

    # For each interior sample x[k]:
    #   a = x[k] - x[k-1]   (slope coming in)
    #   b = x[k] - x[k+1]   (slope going out, but negated)
    # If a and b have the SAME sign, then x[k] is a local extremum:
    #   - both positive: x[k] is higher than both neighbors -> local max
    #   - both negative: x[k] is lower than both neighbors -> local min
    a = x[1:-1] - x[:-2]    # slope segment 1
    b = x[1:-1] - x[2:]     # slope segment 2

    # Same sign means a * b > 0
    is_extremum = (a * b) > 0

    # At least one of the neighbor diffs must exceed the noise floor.
    big_enough = (np.abs(a) >= threshold) | (np.abs(b) >= threshold)

    return np.sum(is_extremum & big_enough, axis=0)


def variance(window):
    """
    Sample variance.

    Note: Phinyomark 2012 showed VAR is highly redundant with MAV for
    zero-mean EMG, so we don't use it in either Path A or Path B.
    It's defined here in case you want to experiment.

    Math:  VAR = (1/(N-1)) * sum((x_i - mean(x))^2)
    """
    # ddof=1 gives the unbiased estimator (divide by N-1, not N).
    return np.var(window, axis=0, ddof=1)


# ---------------------------------------------------------------------
# PATH B: Phinyomark (2012) additional features
# ---------------------------------------------------------------------
# These features are DEFINED here but DISABLED in the pipeline by
# default. See pipeline.py for instructions on enabling them.

def willison_amplitude(window, threshold=1e-5):
    """
    Willison Amplitude / WAMP (Phinyomark Eq. 19).

    Count the number of times the absolute difference between
    consecutive samples exceeds the threshold. Phinyomark notes WAMP
    is "related to the firing of motor unit action potentials"
    because each MUAP produces a sharp local amplitude jump.

    WAMP carries similar information to ZC but is more directly tied
    to the physiological event of a motor unit firing. The 2012 paper
    recommends WAMP over ZC/SSC as the frequency-group representative.

    Threshold tuning: typically 50 microvolts to 100 millivolts at the
    amplified signal level. For your simulator output you'll need to
    experiment. Start with 1e-5 and adjust until rest gives near zero
    and contraction gives tens or hundreds.

    Math:  WAMP = sum( |x_{i+1} - x_i| >= threshold )
    """
    diffs = np.abs(np.diff(window, axis=0))
    return np.sum(diffs >= threshold, axis=0)


def ar_coefficients(window, order=4):
    """
    Autoregressive coefficients via Yule-Walker (Phinyomark Sec 2.1.23).

    The AR model says each sample is a linear combination of the
    previous P samples plus white noise:

        x[i] = a_1*x[i-1] + a_2*x[i-2] + ... + a_P*x[i-P] + noise

    The coefficients a_1, ..., a_P become the features. Unlike MAV
    or WL which summarize AMPLITUDE, AR coefficients summarize the
    TEMPORAL PATTERN: how strongly the current sample depends on the
    recent past. Different muscle activations produce different AR
    signatures because their motor unit firing dynamics differ.

    Phinyomark recommends order=4 based on multiple prior studies.

    Parameters
    ----------
    window : np.ndarray, shape (W,) or (W, C)
    order  : int, AR model order

    Returns
    -------
    coeffs : np.ndarray, shape (order,) for 1D input
                          shape (order, C) for 2D input
    """
    if window.ndim == 1:
        return _ar_single_channel(window, order)

    # Multi-channel: compute per channel and stack column-wise.
    # Result shape is (order, n_channels) so each column is one channel's coeffs.
    return np.stack(
        [_ar_single_channel(window[:, c], order)
         for c in range(window.shape[1])],
        axis=1
    )


def _ar_single_channel(x, order):
    """Private helper: AR coefficients for one 1D signal."""
    # Remove DC offset. AR theory assumes a zero-mean signal.
    # EMG is approximately zero-mean already, but this guards against
    # any baseline drift that may have leaked through preprocessing.
    x = x - np.mean(x)
    n = len(x)

    # Compute the biased autocorrelation at lags 0, 1, 2, ..., order.
    # Autocorrelation at lag k = sum(x[i] * x[i+k]) / n.
    # We use np.dot for efficiency.
    r = np.array([np.dot(x[:n - k], x[k:]) for k in range(order + 1)]) / n

    # Solve the Yule-Walker equations using the Toeplitz structure
    # of the autocorrelation matrix. solve_toeplitz exploits this
    # structure for an O(P^2) solve instead of the O(P^3) of a generic
    # linear solver. For P=4 the speedup is small but it's the
    # idiomatic SciPy way to do this.
    return solve_toeplitz(r[:-1], r[1:])
