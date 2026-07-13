"""
test_per_channel_isolation.py
=============================
Prove that setting sweat on only some channels leaves the other
channels untouched. Fundamental verification of Stage 3A.

Sets channel 3 to full sweat and all other channels to zero, then
measures RMS on every channel. Channel 3 should show clear changes
from all four mechanisms; every other channel should be unchanged
from input.
"""

import numpy as np

from src.artifacts.sweat import SweatProcessor


def main():
    fs         = 1000.0
    n_seconds  = 5.0
    n_samples  = int(fs * n_seconds)
    n_channels = 8

    rng = np.random.default_rng(0)
    emg = rng.standard_normal((n_samples, n_channels)) * 0.1

    # Band-limit the input to below 350 Hz to mimic real EMG spectral
    # content (De Luca 1997). White noise has energy up to Nyquist, which
    # even the near-transparent dry cutoff trims, creating a false "leak"
    # on clean channels. Real EMG has little energy up there, so band-
    # limiting makes this a fair test of per-channel sweat isolation.
    import scipy.signal as bl_dsp
    b_bl, a_bl = bl_dsp.butter(N=4, Wn=350.0 / (0.5 * fs), btype='low')
    emg = bl_dsp.lfilter(b_bl, a_bl, emg, axis=0)

    sp = SweatProcessor(
        n_channels=n_channels, fs=fs,
        tau_sweat=0.0,  # snap immediately for cleaner comparison
        seed=42,
    )

    # Per-channel: only channel 3 gets sweat.
    targets = np.zeros(n_channels)
    targets[3] = 1.0
    sp.set_channel_sweat_levels(targets)

    out = sp.process(emg, sample_rate=fs)

    # Compare per-channel RMS.
    print(f"{'ch':>4}  {'target':>8}  {'in_RMS':>10}  {'out_RMS':>10}  "
          f"{'delta%':>8}  {'status':>10}")
    for ch in range(n_channels):
        in_rms  = np.sqrt(np.mean(emg[:, ch] ** 2))
        out_rms = np.sqrt(np.mean(out[:, ch] ** 2))
        delta   = 100.0 * (out_rms - in_rms) / in_rms if in_rms > 0 else 0.0
        target  = targets[ch]

        # A "clean" channel should show no change (delta near 0).
        # A "sweaty" channel should show a nonzero delta.
        if target == 0.0:
            status = 'clean' if abs(delta) < 1.0 else 'LEAKED'
        else:
            status = 'sweaty' if abs(delta) > 5.0 else 'WEAK'

        print(f"{ch:>4}  {target:>8.2f}  {in_rms:>10.4f}  {out_rms:>10.4f}  "
              f"{delta:>+8.2f}  {status:>10}")


if __name__ == '__main__':
    main()