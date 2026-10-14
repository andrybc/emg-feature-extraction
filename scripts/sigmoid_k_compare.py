"""
sigmoid_k_compare.py
====================
Compare empirical amplitude-vs-sweat curves at multiple sigmoid
steepness values. Runs the actual SweatProcessor at each k, sweeps
sweat level, measures the amplitude multiplier at 30 Hz via FFT,
and plots them together.

This is the concrete answer to "how much does k actually matter."
"""

import numpy as np
import matplotlib.pyplot as plt

from src.artifacts.sweat import SweatProcessor


def measure_multiplier_at_k(k: float, sweat_level: float,
                            fs: float = 1000.0) -> float:
    """Push a 30 Hz probe through a SweatProcessor at a given (k, s)
    and return the output/input FFT magnitude ratio at 30 Hz."""
    rng = np.random.default_rng(0)
    n_samples = int(2.0 * fs)
    emg = rng.standard_normal((n_samples, 8)) * 1e-6

    probe_freq = 30.0
    t = np.arange(n_samples) / fs
    emg[:, 0] += np.sin(2 * np.pi * probe_freq * t)

    sp = SweatProcessor(
        n_channels=8, fs=fs, tau_sweat=0.0, seed=42,
        attenuation_steepness=k,
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
    k_values     = [2.0, 4.0, 8.0, 15.0]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reference linear model.
    ax.plot(sweat_levels, 1.0 - 0.30 * sweat_levels,
            label='pure linear (reference)', linestyle='--',
            color='gray', alpha=0.7)

    print(f"{'s':>6}", end='')
    for k in k_values:
        print(f"  k={k:>4.1f}", end='')
    print()

    all_curves = {}
    for k in k_values:
        curve = np.array([measure_multiplier_at_k(k, s) for s in sweat_levels])
        all_curves[k] = curve
        ax.plot(sweat_levels, curve, marker='o', label=f'sigmoid k = {k}')

    for i, s in enumerate(sweat_levels):
        print(f"{s:6.2f}", end='')
        for k in k_values:
            print(f"  {all_curves[k][i]:6.3f}", end='')
        print()

    ax.axhline(1.0,        color='k', linestyle=':', alpha=0.3)
    ax.axhline(1.0 - 0.30, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Sweat level s')
    ax.set_ylabel('Amplitude multiplier (measured at 30 Hz)')
    ax.set_title('Empirical amplitude vs sweat at different sigmoid steepness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()