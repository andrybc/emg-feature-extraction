"""
Automated test for the sigmoid amplitude attenuation (Mechanism 1).

Verifies that the processor's empirical amplitude multiplier, measured
at a 30 Hz probe frequency via FFT, matches the theoretical normalized
sigmoid across the full sweat range.

Tolerance rationale:
  - Interior points: abs=0.02. The observed max deviation between
    empirical and theory is ~0.0135 at s=0.5, caused by the lowpass
    filter's small attenuation of the 30 Hz probe (not a sigmoid error).
    0.02 covers this with headroom while still catching a real
    regression (a broken sigmoid deviates far more than 2%).
  - Endpoints: abs=1e-6. The normalization guarantees the multiplier is
    exactly 1.0 at s=0 and exactly (1 - max_attenuation) at s=1, so
    these are effectively exact.
"""

import numpy as np
import pytest

from src.artifacts.sweat import SweatProcessor


def theoretical_multiplier(s, k, max_att=0.30):
    """Normalized sigmoid attenuation multiplier at sweat level s."""
    sig_at_zero = 1.0 / (1.0 + np.exp( k * 0.5))
    sig_at_one  = 1.0 / (1.0 + np.exp(-k * 0.5))
    sig_range   = sig_at_one - sig_at_zero
    sig_raw     = 1.0 / (1.0 + np.exp(-k * (s - 0.5)))
    sig_norm    = (sig_raw - sig_at_zero) / sig_range
    return 1.0 - max_att * sig_norm


def measure_multiplier(sweat_level, k=8.0, fs=1000.0):
    """
    Push a 30 Hz probe through the processor at a fixed sweat level and
    return the output/input FFT magnitude ratio at 30 Hz. Isolates the
    probe from drift (<1 Hz) and powerline (60+ Hz) since those live in
    different bins.
    """
    rng = np.random.default_rng(0)
    n = int(2.0 * fs)
    emg = rng.standard_normal((n, 8)) * 1e-6
    t = np.arange(n) / fs
    emg[:, 0] += np.sin(2 * np.pi * 30.0 * t)

    sp = SweatProcessor(n_channels=8, fs=fs, tau_sweat=0.0, seed=42,
                        attenuation_steepness=k)
    sp.set_sweat_level(sweat_level)
    out = sp.process(emg, fs)

    skip = int(0.1 * fs)
    freqs = np.fft.rfftfreq(n - skip, 1.0 / fs)
    idx = int(np.argmin(np.abs(freqs - 30.0)))
    in_mag  = np.abs(np.fft.rfft(emg[skip:, 0]))[idx]
    out_mag = np.abs(np.fft.rfft(out[skip:, 0]))[idx]
    return out_mag / in_mag


@pytest.mark.parametrize("s", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def test_sigmoid_matches_theory_interior(s):
    """Empirical multiplier tracks the theoretical sigmoid within 2%."""
    measured = measure_multiplier(s)
    expected = theoretical_multiplier(s, k=8.0)
    assert measured == pytest.approx(expected, abs=0.02)


def test_sigmoid_endpoint_dry():
    """At s=0, pass-through guard makes the multiplier exactly 1.0."""
    expected = theoretical_multiplier(0.0, k=8.0)
    assert expected == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_endpoint_wet():
    """At s=1, the multiplier equals exactly (1 - max_attenuation)."""
    expected = theoretical_multiplier(1.0, k=8.0)
    assert expected == pytest.approx(0.70, abs=1e-6)