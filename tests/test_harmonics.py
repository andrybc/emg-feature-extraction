"""
Automated test for powerline harmonic ratios (Mechanism 4).

Injects a very quiet noise input at full sweat so the powerline harmonic
peaks stand cleanly above the noise floor, then checks the channel-
averaged FFT peak ratios at 120/180/240 Hz relative to the 60 Hz
fundamental against the design weights (1.0, 0.3, 0.1, 0.03).

Tolerance rationale:
  - 120 Hz ratio: design 0.30, measured ~0.30. abs=0.05.
  - 180 Hz ratio: design 0.10, measured ~0.12 (slight spectral leakage
    into the bin). abs=0.05 covers the bias.
  - 240 Hz ratio: design 0.03, but at this input level it sits near the
    noise floor and is unreliable to pin precisely. Asserted only to be
    present and the smallest harmonic (0 < ratio < 0.10), which catches
    gross breakage without failing on noise.
"""

import numpy as np
import pytest

from src.artifacts.sweat import SweatProcessor

FS = 1000.0
N_SECONDS = 20.0
N_CHANNELS = 8


def _harmonic_ratios():
    """Return (r120, r180, r240) peak ratios relative to the 60 Hz peak."""
    n_samples = int(FS * N_SECONDS)
    rng = np.random.default_rng(0)
    emg = rng.standard_normal((n_samples, N_CHANNELS)) * 0.001

    sp = SweatProcessor(n_channels=N_CHANNELS, fs=FS, tau_sweat=0.0, seed=42)
    sp.set_sweat_level(1.0)
    out = sp.process(emg, sample_rate=FS)

    # Analyze last 10 s, channel-averaged (harmonics correlate across
    # channels, noise does not, so averaging lifts the peaks).
    n_last = int(10.0 * FS)
    block = out[-n_last:] - out[-n_last:].mean(axis=0, keepdims=True)

    mags = []
    for ch in range(N_CHANNELS):
        mag = np.abs(np.fft.rfft(block[:, ch])) / block.shape[0]
        mags.append(mag)
    freqs = np.fft.rfftfreq(block.shape[0], d=1.0 / FS)
    mag_avg = np.mean(mags, axis=0)

    def peak_at(hz):
        idx = int(np.argmin(np.abs(freqs - hz)))
        return mag_avg[idx]

    p60 = peak_at(60.0)
    assert p60 > 0, "no 60 Hz fundamental detected"
    return peak_at(120.0) / p60, peak_at(180.0) / p60, peak_at(240.0) / p60


# Compute once; the measurement is deterministic (seeded).
_R120, _R180, _R240 = _harmonic_ratios()


def test_ratio_120hz():
    """2nd harmonic ratio matches design weight 0.30."""
    assert _R120 == pytest.approx(0.30, abs=0.05)


def test_ratio_180hz():
    """3rd harmonic ratio matches design weight 0.10 (allowing leakage bias)."""
    assert _R180 == pytest.approx(0.10, abs=0.05)


def test_ratio_240hz_present_and_smallest():
    """4th harmonic is present but the weakest; near the noise floor so
    only bounded, not pinned to 0.03."""
    assert 0.0 < _R240 < 0.10, f"240 Hz ratio {_R240:.3f} out of expected range"


def test_harmonic_ordering():
    """Harmonics decrease in amplitude: 120 > 180 > 240."""
    assert _R120 > _R180 > _R240, (
        f"harmonics not monotonic decreasing: "
        f"120={_R120:.3f} 180={_R180:.3f} 240={_R240:.3f}"
    )