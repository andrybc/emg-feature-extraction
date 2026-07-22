"""
test_mechanism_1_isolated.py
============================
Isolate Mechanism 1 (sigmoid amplitude attenuation) by zeroing the
other three sweat mechanisms. Feed a known sine wave in, measure
what comes out at each sweat level, and confirm the amplitude
multiplier matches the sigmoid math.

This is the "did the sigmoid feel wrong in the simulator, or was it
the other mechanisms" test. If Mechanism 1 in isolation matches theory,
then whatever aggressiveness you're feeling in the live simulator is
from Mechanisms 2, 3, or 4, not from the amplitude attenuation.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.artifacts.sweat import SweatProcessor


def theoretical_multiplier(s: float, k: float, max_att: float = 0.30) -> float:
    """The sigmoid attenuation math, reproduced in isolation for comparison."""
    sig_at_zero = 1.0 / (1.0 + np.exp( k * 0.5))
    sig_at_one  = 1.0 / (1.0 + np.exp(-k * 0.5))
    sig_range   = sig_at_one - sig_at_zero
    sig_raw     = 1.0 / (1.0 + np.exp(-k * (s - 0.5)))
    sig_norm    = (sig_raw - sig_at_zero) / sig_range
    return 1.0 - max_att * sig_norm


def measure_isolated(sweat_level: float, k: float = 8.0,
                     fs: float = 1000.0) -> float:
    """
    Push a clean 30 Hz sine through a SweatProcessor with mechanisms
    2, 3, and 4 disabled. Return the output/input amplitude ratio
    measured at 30 Hz.

    Mechanism 2 is disabled by setting wet_cutoff_hz == dry_cutoff_hz,
    so the lowpass cutoff never moves regardless of sweat level. The
    lowpass is still active but at a fixed 400 Hz, which lets the 30 Hz
    probe pass through untouched.

    Mechanism 3 is disabled by setting drift_scale = 0.0.

    Mechanism 4 is disabled by setting imbalance_range = (0.0, 0.0),
    so per-channel imbalance is exactly zero on every channel, and
    the powerline amplitude is 0.
    """
    rng = np.random.default_rng(0)
    n_samples = int(2.0 * fs)
    emg = rng.standard_normal((n_samples, 8)) * 1e-6

    probe_freq = 30.0
    t = np.arange(n_samples) / fs
    emg[:, 0] += np.sin(2 * np.pi * probe_freq * t)

    sp = SweatProcessor(
        n_channels=8,
        fs=fs,
        tau_sweat=0.0,
        seed=42,
        attenuation_steepness=k,
        wet_cutoff_hz=400.0,         # == dry cutoff, so no cutoff change
        drift_scale=0.0,             # kill Mechanism 3
        imbalance_range=(0.0, 0.0),  # kill Mechanism 4
    )
    sp.set_sweat_level(sweat_level)
    out = sp.process(emg, sample_rate=fs)

    skip = int(0.1 * fs)
    x_in  = emg[skip:, 0]
    x_out = out[skip:, 0]

    freqs   = np.fft.rfftfreq(len(x_in), 1.0 / fs)
    idx     = int(np.argmin(np.abs(freqs - probe_freq)))
    in_mag  = np.abs(np.fft.rfft(x_in))[idx]
    out_mag = np.abs(np.fft.rfft(x_out))[idx]

    return out_mag / in_mag


def main():
    sweat_levels = np.linspace(0.0, 1.0, 11)
    k            = 8.0

    print(f"Isolated Mechanism 1 test (k = {k}):")
    print(f"{'s':>6}  {'measured':>10}  {'theory':>10}  {'diff':>10}")
    measured = []
    for s in sweat_levels:
        m = measure_isolated(s, k)
        t = theoretical_multiplier(s, k)
        measured.append(m)
        print(f"{s:6.2f}  {m:10.4f}  {t:10.4f}  {m - t:+10.4f}")

    measured = np.array(measured)
    theory   = np.array([theoretical_multiplier(s, k) for s in sweat_levels])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweat_levels, theory,   label='theoretical sigmoid', color='blue')
    ax.scatter(sweat_levels, measured, color='red', s=40, zorder=5,
               label='measured (Mechanisms 2, 3, 4 disabled)')
    ax.set_xlabel('Sweat level s')
    ax.set_ylabel('Amplitude multiplier at 30 Hz')
    ax.set_title('Mechanism 1 isolated: does the sigmoid work by itself?')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()