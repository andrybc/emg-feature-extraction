"""
Quick FFT check of a sweat-processed CSV.

Compares the spectrum of one channel in the clean and sweaty recordings.
Look for spikes at 60, 120, 180, 240 Hz in the sweaty spectrum that are
absent (or much smaller) in the clean spectrum.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def one_sided_fft(x: np.ndarray, fs: float):
    """
    Return (freqs_Hz, magnitude) for a real-valued signal.

    Uses rfft which is the efficient real-input FFT: returns only the
    positive-frequency half of the spectrum (the negative half is a
    mirror image for real signals, so it carries no new information).
    """
    n     = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag   = np.abs(np.fft.rfft(x))
    # Normalize so amplitudes are independent of window length.
    mag   = mag / n
    return freqs, mag


def main(clean_csv: str, sweaty_csv: str, channel: int = 1):
    clean  = pd.read_csv(clean_csv)
    sweaty = pd.read_csv(sweaty_csv)

    col = f'ch{channel}_mV'
    x_clean  = clean[col].values
    x_sweaty = sweaty[col].values

    dt = np.diff(clean['timestamp_s'].values)
    fs = 1.0 / np.median(dt)
    print(f"Detected fs = {fs:.1f} Hz")

    n_last = int(5.0 * fs)
    x_clean  = x_clean[-n_last:]
    x_sweaty = x_sweaty[-n_last:]

    # Remove DC offset so it doesn't dominate the FFT plot.
    # Subtracting the mean is the standard cleanup before FFT.
    x_clean  = x_clean  - x_clean.mean()
    x_sweaty = x_sweaty - x_sweaty.mean()

    f_clean,  mag_clean  = one_sided_fft(x_clean,  fs)
    f_sweaty, mag_sweaty = one_sided_fft(x_sweaty, fs)

    # Skip the 0 Hz bin. It still carries the drift energy even after
    # mean subtraction because drift is not perfectly DC.
    f_clean,  mag_clean  = f_clean[1:],  mag_clean[1:]
    f_sweaty, mag_sweaty = f_sweaty[1:], mag_sweaty[1:]

    # Print peak magnitudes at each harmonic. Read the FFT bin closest
    # to each target frequency.
    print("\nHarmonic peak magnitudes (sweaty channel):")
    peak_60 = None
    for k in (1, 2, 3, 4):
        f_target = 60.0 * k
        idx      = int(np.argmin(np.abs(f_sweaty - f_target)))
        peak     = mag_sweaty[idx]
        if k == 1:
            peak_60 = peak
            print(f"  {f_target:5.0f} Hz : {peak:.4e}  (reference, weight 1.00)")
        else:
            expected_weight = [0.3, 0.1, 0.03][k - 2]
            actual_ratio    = peak / peak_60 if peak_60 > 0 else 0
            print(f"  {f_target:5.0f} Hz : {peak:.4e}  "
                  f"ratio {actual_ratio:.3f} (expected ~{expected_weight})")

    # Plot with log y-scale so all four peaks are readable.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f_clean,  mag_clean,  label='clean',    alpha=0.5)
    ax.semilogy(f_sweaty, mag_sweaty, label='sweat 1.0', alpha=0.8)

    for k in (1, 2, 3, 4):
        ax.axvline(60.0 * k, color='red', linestyle=':', alpha=0.4,
                   label='powerline harmonic' if k == 1 else None)

    ax.set_xlim(0, 300)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|FFT| / N  (log scale)')
    ax.set_title(f'Spectrum of {col}, last {n_last/fs:.1f}s')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python fft_check.py CLEAN.csv SWEATY.csv [channel]")
        sys.exit(1)

    channel = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    main(sys.argv[1], sys.argv[2], channel)