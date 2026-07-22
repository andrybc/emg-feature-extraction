"""
Synthetic-input harmonic verification.

Uses a low-amplitude Gaussian noise input instead of real EMG. This
removes the muscle-spectrum contamination that makes the harmonic
ratios noisy on real recordings.

Expected result: the four harmonic peaks stand cleanly above a
featureless noise floor, and their ratios match the weights closely.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.artifacts.sweat import SweatProcessor


def one_sided_fft(x, fs):
    n     = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag   = np.abs(np.fft.rfft(x)) / n
    return freqs, mag


def main():
    fs         = 1000.0
    n_seconds  = 20.0                    # longer window, tighter FFT bins
    n_samples  = int(fs * n_seconds)
    n_channels = 8

    # Synthetic input: very quiet noise. The powerline amplitude in
    # SweatProcessor scales with running_rms, so a small input RMS
    # gives a small powerline signal, but the noise floor drops even
    # faster, giving us headroom to see the harmonics.
    rng = np.random.default_rng(0)
    emg = rng.standard_normal((n_samples, n_channels)) * 0.001

    sp = SweatProcessor(n_channels=n_channels, fs=fs, tau_sweat=0.0, seed=42)
    sp.set_sweat_level(1.0)
    out = sp.process(emg, sample_rate=fs)

    # Analyze last 10 seconds, averaged across all 8 channels. The
    # powerline harmonics are correlated across channels (same sine
    # values, only amplitudes differ), so averaging FFT magnitudes
    # boosts them relative to per-channel noise floors.
    n_last = int(10.0 * fs)
    block  = out[-n_last:] - out[-n_last:].mean(axis=0, keepdims=True)

    mags = []
    for ch in range(n_channels):
        f, mag = one_sided_fft(block[:, ch], fs)
        mags.append(mag)
    mag_avg = np.mean(mags, axis=0)
    f       = f[1:]
    mag_avg = mag_avg[1:]

    print("Harmonic peak magnitudes (channel-averaged):")
    peak_60 = None
    for k in (1, 2, 3, 4):
        f_target = 60.0 * k
        idx      = int(np.argmin(np.abs(f - f_target)))
        peak     = mag_avg[idx]
        if k == 1:
            peak_60 = peak
            print(f"  {f_target:5.0f} Hz : {peak:.4e}  (reference)")
        else:
            expected = [0.3, 0.1, 0.03][k - 2]
            ratio    = peak / peak_60 if peak_60 > 0 else 0
            print(f"  {f_target:5.0f} Hz : {peak:.4e}  "
                  f"ratio {ratio:.3f} (expected ~{expected})")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f, mag_avg, label='channel-averaged spectrum')
    for k in (1, 2, 3, 4):
        ax.axvline(60.0 * k, color='red', linestyle=':', alpha=0.4,
                   label='powerline harmonic' if k == 1 else None)
    ax.set_xlim(0, 300)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|FFT| / N  (log scale)')
    ax.set_title('Synthetic-input harmonic spectrum, channel-averaged')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()