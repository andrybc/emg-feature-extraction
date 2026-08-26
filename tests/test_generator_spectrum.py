"""
Automated test that the EMG signal generator's output spectrum stays
fixed in absolute Hz regardless of sample rate.

Before the fs-invariance fix, the shaping filter used hardcoded
coefficients whose passband scaled with Nyquist, leaking energy above
400 Hz at high fs (which the sweat low-pass then removed, making 1%
sweat halve the signal at fs>=2000). After the fix (4th-order Butterworth
bandpass designed at runtime), the energy stays in ~20-350 Hz at every
rate.

Reproducibility: the generator draws from NumPy's global RNG, so
np.random.seed is set before each measurement.

Tolerances from measured post-fix values (centroid 186/191/198 Hz,
above-400 energy 0.1/2.3/3.5% at fs=1000/2000/4000):
  - Centroid within 175-215 Hz at every rate. A regression where the
    spectrum scales with Nyquist would push the high-fs centroid far
    outside this band.
  - Above-400 Hz energy under 6% at every rate. Catches the original
    leak (which reached 13% at fs=4000 before the fix).
"""

import numpy as np
import importlib.util
import pytest

_spec = importlib.util.spec_from_file_location("emg_simulator", "emg_simulator.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EMGSignalGenerator = _mod.EMGSignalGenerator

RATES = [1000.0, 2000.0, 4000.0]


def _spectrum_stats(fs):
    """Return (frac_above_400, centroid_hz) for the generator at this fs."""
    np.random.seed(4321)
    gen = EMGSignalGenerator(fs=fs)
    gen.set_gesture('Hand Closed')
    n = int(fs * 4)
    samples, _ = gen.generate_samples(n_samples=n)
    x = samples[:, 0]
    x = x - x.mean()

    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    power = np.abs(np.fft.rfft(x)) ** 2
    total = power.sum()

    frac_above = power[freqs >= 400].sum() / total
    centroid = (freqs * power).sum() / total
    return frac_above, centroid


@pytest.mark.parametrize("fs", RATES)
def test_centroid_stable(fs):
    """Spectral centroid stays in the EMG band regardless of sample rate."""
    _, centroid = _spectrum_stats(fs)
    assert 175.0 <= centroid <= 215.0, (
        f"fs={fs}: centroid {centroid:.1f} Hz outside 175-215 Hz "
        f"(spectrum may be scaling with Nyquist)"
    )


@pytest.mark.parametrize("fs", RATES)
def test_energy_stays_in_band(fs):
    """Energy above 400 Hz stays small regardless of sample rate."""
    frac_above, _ = _spectrum_stats(fs)
    assert frac_above < 0.06, (
        f"fs={fs}: {100*frac_above:.1f}% of energy above 400 Hz "
        f"(exceeds 6%; the shaping filter may be leaking at high fs)"
    )


def test_centroid_invariance_across_rates():
    """The centroid does not drift much across sample rates."""
    centroids = [_spectrum_stats(fs)[1] for fs in RATES]
    drift = max(centroids) - min(centroids)
    assert drift < 30.0, (
        f"centroid drifts {drift:.1f} Hz across sample rates "
        f"({[round(c, 1) for c in centroids]}); expected < 30 Hz"
    )