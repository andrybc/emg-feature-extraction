"""
Automated test for sample-rate invariance of the sweat processor.

For each sweat level, the amplitude ratio (output RMS / input RMS) is
measured across fs = 1000/2000/4000 on EMG-representative generator
output. The spread (max - min, relative to max) must stay within a
per-level tolerance.

Reproducibility: the generator's internal noise draws from NumPy's global
RNG, so np.random.seed(12345) is set before each measurement to make the
spreads identical run to run.

Tolerance rationale (see notes/sweat_model_provenance.md):
Invariance is tight at low-to-mid sweat (3-4%) and widens at full sweat
(~9%). The full-sweat residual decomposes into ~2.6% from the generator's
high-frequency tail through the wet low-pass, plus drift-mechanism spread
that is largely short-window estimation noise on a 0.5 Hz process. Both
are documented floors, not defects. Tolerances are the measured spread
plus ~2 points of margin to absorb seed/platform variation.
"""

import numpy as np
import importlib.util
import pytest

from src.artifacts.sweat import SweatProcessor

_spec = importlib.util.spec_from_file_location("emg_simulator", "emg_simulator.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EMGSignalGenerator = _mod.EMGSignalGenerator

RATES = [1000.0, 2000.0, 4000.0]

# (sweat_level, tolerance_fraction) from measured reproducible spreads
# (generator seeded) plus ~2 points of margin.
LEVEL_TOLERANCE = [
    (0.01, 0.05),
    (0.10, 0.05),
    (0.25, 0.06),
    (0.50, 0.05),
    (0.75, 0.08),
    (1.00, 0.12),
]


def amplitude_ratio(fs, sweat_level, seconds=4.0):
    """Ratio of output RMS to input RMS at one fs and sweat level.

    Seeds NumPy's global RNG so the generator's internal noise is
    deterministic and the measurement is reproducible run to run.
    """
    np.random.seed(12345)
    gen = EMGSignalGenerator(fs=fs)
    gen.set_gesture('Hand Closed')
    n = int(fs * seconds)
    clean, _ = gen.generate_samples(n_samples=n)

    sp = SweatProcessor(n_channels=8, fs=fs, tau_sweat=0.0, seed=42)
    sp.set_sweat_level(sweat_level)
    out = sp.process(clean.copy(), fs)

    return np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(clean ** 2))


@pytest.mark.parametrize("sweat_level,tolerance", LEVEL_TOLERANCE)
def test_fs_invariance_per_level(sweat_level, tolerance):
    """Amplitude ratio spread across sample rates stays within tolerance."""
    ratios = [amplitude_ratio(fs, sweat_level) for fs in RATES]
    spread = (max(ratios) - min(ratios)) / max(ratios)
    assert spread <= tolerance, (
        f"sweat={sweat_level}: spread {spread:.1%} exceeds tolerance "
        f"{tolerance:.0%}; ratios={[round(r, 3) for r in ratios]}"
    )


def test_low_sweat_amplitude_near_unity():
    """At 1% sweat the amplitude is nearly untouched at every rate.

    This is the original high-fs symptom that motivated the generator
    fix: before the fix, 1% sweat slashed amplitude at high fs. It should
    now stay above 0.95 at all rates.
    """
    for fs in RATES:
        ratio = amplitude_ratio(fs, 0.01)
        assert ratio >= 0.95, f"fs={fs}: 1% sweat ratio {ratio:.3f} below 0.95"