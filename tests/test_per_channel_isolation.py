"""
Automated per-channel sweat isolation test (Stage 3A).

Sets one channel to full sweat and all others to zero, then checks that
the dry channels are essentially unchanged (no leak) while the sweaty
channel is clearly affected by all four mechanisms.

The input is band-limited to 350 Hz to mimic real EMG spectral content
(De Luca 1997). White noise has energy up to Nyquist, which even the
near-transparent dry cutoff trims, creating a false "leak" on clean
channels. Real EMG has little energy there, so band-limiting makes this
a fair test of isolation rather than of the dry low-pass.

Tolerances (matching the original diagnostic thresholds):
  - Dry channels: |delta RMS| < 1%. When any channel is sweaty the whole
    buffer runs the pipeline, but on a dry channel attenuation=1.0,
    drift and powerline scale to 0, and the dry cutoff is near-transparent
    on band-limited input, so the change is sub-1%.
  - Sweaty channel: |delta RMS| > 5%. All four mechanisms act, so the
    change is clearly nonzero.
"""

import numpy as np
import scipy.signal as dsp
import pytest

from src.artifacts.sweat import SweatProcessor

FS = 1000.0
N_CHANNELS = 8
SWEATY_CH = 3


def _deltas():
    """Return per-channel RMS delta (%) with only SWEATY_CH sweated."""
    n_samples = int(FS * 5.0)
    rng = np.random.default_rng(0)
    emg = rng.standard_normal((n_samples, N_CHANNELS)) * 0.1

    # Band-limit to 350 Hz to mimic real EMG (see module docstring).
    b_bl, a_bl = dsp.butter(N=4, Wn=350.0 / (0.5 * FS), btype='low')
    emg = dsp.lfilter(b_bl, a_bl, emg, axis=0)

    sp = SweatProcessor(n_channels=N_CHANNELS, fs=FS, tau_sweat=0.0, seed=42)
    targets = np.zeros(N_CHANNELS)
    targets[SWEATY_CH] = 1.0
    sp.set_channel_sweat_levels(targets)
    out = sp.process(emg, sample_rate=FS)

    deltas = np.zeros(N_CHANNELS)
    for ch in range(N_CHANNELS):
        in_rms = np.sqrt(np.mean(emg[:, ch] ** 2))
        out_rms = np.sqrt(np.mean(out[:, ch] ** 2))
        deltas[ch] = 100.0 * (out_rms - in_rms) / in_rms if in_rms > 0 else 0.0
    return deltas


# Deterministic (seeded); compute once.
_DELTAS = _deltas()


@pytest.mark.parametrize("ch", [c for c in range(N_CHANNELS) if c != SWEATY_CH])
def test_dry_channel_no_leak(ch):
    """Each dry channel changes by less than 1% (no cross-channel leak)."""
    assert abs(_DELTAS[ch]) < 1.0, (
        f"channel {ch} (dry) leaked: RMS delta {_DELTAS[ch]:+.2f}% "
        f"(sweat was applied only to channel {SWEATY_CH})"
    )


def test_sweaty_channel_affected():
    """The sweated channel changes by more than 5%."""
    assert abs(_DELTAS[SWEATY_CH]) > 5.0, (
        f"channel {SWEATY_CH} (sweaty) barely changed: "
        f"RMS delta {_DELTAS[SWEATY_CH]:+.2f}%"
    )