"""
test_sweat_fs_stability.py
==========================
Acceptance test for the fs-invariant generator fix.

The original bug: at high sample rates, even 1% sweat appeared to slash
the signal amplitude, because the generator leaked high-frequency energy
that the sweat lowpass then removed. This test measures the amplitude
ratio (sweaty RMS / clean RMS) at 1% sweat across sample rates, using
the actual simulator generator feeding the actual sweat processor.

Pass criterion: the 1% ratio should be close to 1.0 (barely any effect)
and roughly the same at every sample rate. Before the fix, the ratio
dropped steeply as fs increased.
"""

import numpy as np
import importlib.util

from src.artifacts.sweat import SweatProcessor

# Load the generator from the simulator module.
spec = importlib.util.spec_from_file_location("emg_simulator", "emg_simulator.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
EMGSignalGenerator = mod.EMGSignalGenerator


def measure_ratio(fs, sweat_level, seconds=4.0):
    """
    Generate clean EMG at the given fs, then process a copy through the
    sweat processor at the given sweat level. Return sweaty/clean RMS.
    """
    gen = EMGSignalGenerator(fs=fs)
    gen.set_gesture('Hand Closed')
    n = int(fs * seconds)
    clean, _ = gen.generate_samples(n_samples=n)

    sp = SweatProcessor(n_channels=8, fs=fs, tau_sweat=0.0, seed=42)
    sp.set_sweat_level(sweat_level)
    sweaty = sp.process(clean.copy(), fs)

    clean_rms  = np.sqrt(np.mean(clean ** 2))
    sweaty_rms = np.sqrt(np.mean(sweaty ** 2))
    return sweaty_rms / clean_rms


def main():
    print("Amplitude ratio (sweaty RMS / clean RMS) vs sample rate\n")
    print(f"{'fs':>6}  {'1% sweat':>10}  {'50% sweat':>10}  {'100% sweat':>11}")
    for fs in [1000.0, 2000.0, 4000.0]:
        r01  = measure_ratio(fs, 0.01)
        r50  = measure_ratio(fs, 0.50)
        r100 = measure_ratio(fs, 1.00)
        print(f"{fs:>6.0f}  {r01:>10.3f}  {r50:>10.3f}  {r100:>11.3f}")

    print("\nInterpretation:")
    print("  1% sweat column should be ~0.97-1.00 at ALL sample rates.")
    print("  If it drops with fs (e.g. 0.95 -> 0.75 -> 0.60), the bug is back.")
    print("  50% and 100% columns should show real attenuation but stay")
    print("  roughly consistent across sample rates.")


if __name__ == '__main__':
    main()