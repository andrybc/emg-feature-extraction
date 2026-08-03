"""
test_dry_passthrough.py
=======================
Verify the sweat processor's bit-exact pass-through guarantee: when
every channel is dry (target and current interface state both zero),
process() returns the input completely unchanged.

This is stronger than the per-channel isolation test, which checks that
dry channels stay within measurement noise. Here we assert the output
is numerically identical to the input, confirming the pass-through
guard bypasses all four mechanisms when there is no sweat anywhere.
"""

import numpy as np

from src.artifacts.sweat import SweatProcessor


def test_bit_exact_passthrough():
    fs = 1000.0
    sp = SweatProcessor(n_channels=8, fs=fs, seed=42)
    # No sweat set: all channels dry from construction.
    rng = np.random.default_rng(0)
    emg = rng.standard_normal((500, 8)) * 0.1

    out = sp.process(emg, fs)

    # The guard returns the same array object, untouched.
    assert np.array_equal(out, emg), "dry pass-through altered the signal"
    print("PASS: output is numerically identical to input when all channels dry")


def test_passthrough_after_drying():
    """
    After sweating then returning to fully dry (target zero, current
    decayed below threshold), pass-through should re-engage.
    """
    fs = 1000.0
    sp = SweatProcessor(n_channels=8, fs=fs, tau_sweat=0.0, seed=42)
    rng = np.random.default_rng(1)

    # Sweat up, process a buffer (mechanisms active).
    sp.set_sweat_level(0.8)
    emg1 = rng.standard_normal((500, 8)) * 0.1
    out1 = sp.process(emg1, fs)
    assert not np.array_equal(out1, emg1), "sweat had no effect (unexpected)"

    # Dry fully. With tau_sweat=0 the current state snaps to zero.
    sp.set_sweat_level(0.0)
    emg2 = rng.standard_normal((500, 8)) * 0.1
    out2 = sp.process(emg2, fs)
    assert np.array_equal(out2, emg2), "pass-through did not re-engage after drying"
    print("PASS: pass-through re-engages after sweating then drying")


if __name__ == '__main__':
    test_bit_exact_passthrough()
    test_passthrough_after_drying()
    print("\nAll dry pass-through tests passed.")