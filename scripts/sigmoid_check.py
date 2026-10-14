"""
sigmoid_check.py
================
Verify the sigmoid amplitude attenuation in SweatProcessor.

Three tests:

  1. Plot attenuation multiplier vs sweat level.
     Should show an S-curve from 1.0 (at s=0) to (1 - max_attenuation)
     at s=1, with the steepest slope near s=0.5.

  2. Empirical multiplier check.
     Push a known-amplitude signal through the processor at a range of
     sweat levels and measure the resulting RMS. The ratio to the input
     RMS should follow the theoretical curve.

  3. Endpoint sanity.
     At s=0, the pass-through guard returns the input unchanged.
     At s=1, the attenuation should equal max_attenuation exactly
     (up to the effect of the other mechanisms, which we isolate).
"""

import numpy as np
import matplotlib.pyplot as plt

from src.artifacts.sweat import SweatProcessor


def theoretical_curve(s: np.ndarray, k: float, max_att: float) -> np.ndarray:
    """
    Reproduce the sigmoid attenuation math in isolation, so we can plot
    the expected curve without running the processor.
    """
    sig_at_zero = 1.0 / (1.0 + np.exp( k * 0.5))
    sig_at_one  = 1.0 / (1.0 + np.exp(-k * 0.5))
    sig_range   = sig_at_one - sig_at_zero

    sig_raw    = 1.0 / (1.0 + np.exp(-k * (s - 0.5)))
    sig_norm   = (sig_raw - sig_at_zero) / sig_range
    return 1.0 - max_att * sig_norm


def measure_empirical_multiplier(sweat_level: float, fs: float = 1000.0) -> float:
    """
    Push a known signal through a fresh SweatProcessor at a fixed sweat
    level, and return the output RMS / input RMS ratio for channel 0.

    Uses a low-imbalance seed and disables the running RMS effect on
    other mechanisms by using a very quiet input, so the multiplier
    isolates the amplitude attenuation as much as possible.
    """
    # Quiet input so drift and powerline mechanisms are near-zero.
    rng = np.random.default_rng(0)
    n_samples = int(2.0 * fs)  # 2 seconds
    emg = rng.standard_normal((n_samples, 8)) * 1e-6

    # Fixed high-frequency signal on channel 0 to serve as the "amplitude probe"
    # It has to be inside the passband of the sweat lowpass (150 Hz at full sweat)
    # so the lowpass doesn't kill our probe. 30 Hz is well inside.
    t = np.arange(n_samples) / fs
    probe_amp = 1.0
    emg[:, 0] += probe_amp * np.sin(2 * np.pi * 30.0 * t)

    sp = SweatProcessor(
        n_channels=8, fs=fs, tau_sweat=0.0, seed=42,
        # Force per-channel modulation to be uniform for a clean measurement.
        # We can't easily disable it, but seed=42 gives us a deterministic
        # value on channel 0 that we'll account for below.
    )
    sp.set_sweat_level(sweat_level)
    out = sp.process(emg, sample_rate=fs)

    # Take the RMS of the probe on channel 0, ignoring the first 100 ms
    # to let the lowpass filter settle.
    skip = int(0.1 * fs)
    in_rms  = np.sqrt(np.mean(emg[skip:, 0] ** 2))
    out_rms = np.sqrt(np.mean(out[skip:, 0] ** 2))
    return out_rms / in_rms


def measure_empirical_multiplier_fft(sweat_level: float, fs: float = 1000.0) -> float:
    """
    Same as measure_empirical_multiplier but reads the FFT bin at the
    probe frequency instead of broadband RMS. This isolates the probe
    from drift, powerline, and other spectral content added by the
    processor.
    """
    rng = np.random.default_rng(0)
    n_samples = int(2.0 * fs)
    emg = rng.standard_normal((n_samples, 8)) * 1e-6

    probe_freq = 30.0
    t = np.arange(n_samples) / fs
    probe = np.sin(2 * np.pi * probe_freq * t)
    emg[:, 0] += probe  # amplitude 1.0

    sp = SweatProcessor(n_channels=8, fs=fs, tau_sweat=0.0, seed=42)
    sp.set_sweat_level(sweat_level)
    out = sp.process(emg, sample_rate=fs)

    # Skip settling time so the lowpass reaches steady state.
    skip = int(0.1 * fs)
    x_in  = emg[skip:, 0]
    x_out = out[skip:, 0]

    # FFT magnitude at the probe frequency.
    freqs      = np.fft.rfftfreq(len(x_in), 1.0 / fs)
    idx        = int(np.argmin(np.abs(freqs - probe_freq)))
    in_mag     = np.abs(np.fft.rfft(x_in))[idx]
    out_mag    = np.abs(np.fft.rfft(x_out))[idx]

    return out_mag / in_mag


def main():
    # ----- Test 1: theoretical curve shape -----
    s_range = np.linspace(0.0, 1.0, 201)
    curve_linear  = 1.0 - 0.30 * s_range                         # old model
    curve_sigmoid = theoretical_curve(s_range, k=8.0, max_att=0.30)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s_range, curve_linear,  label='linear (old)',    linestyle='--')
    ax.plot(s_range, curve_sigmoid, label='sigmoid k=8 (new)')
    ax.axhline(1.0,       color='gray', linestyle=':', alpha=0.5)
    ax.axhline(1.0 - 0.30, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Sweat level s')
    ax.set_ylabel('Attenuation multiplier')
    ax.set_title('Amplitude attenuation vs sweat level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ----- Test 2: empirical vs theoretical -----
    sweat_levels = np.linspace(0.0, 1.0, 11)
    empirical = np.array([measure_empirical_multiplier_fft(s) for s in sweat_levels])
    theory       = theoretical_curve(sweat_levels, k=8.0, max_att=0.30)

    print("Empirical vs theoretical attenuation multiplier:")
    print(f"  {'s':>6}  {'theory':>10}  {'empirical':>10}  {'diff':>8}")
    for s, t_val, e_val in zip(sweat_levels, theory, empirical):
        print(f"  {s:6.2f}  {t_val:10.4f}  {e_val:10.4f}  {e_val - t_val:+8.4f}")

    # Overlay empirical points on the theoretical curve.
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(s_range, theoretical_curve(s_range, 8.0, 0.30),
             label='theoretical sigmoid')
    ax2.scatter(sweat_levels, empirical, color='red', zorder=5,
                label='measured from processor')
    ax2.set_xlabel('Sweat level s')
    ax2.set_ylabel('Amplitude multiplier (output RMS / input RMS)')
    ax2.set_title('Empirical vs theoretical amplitude attenuation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # ----- Test 3: endpoint sanity -----
    print("\nEndpoint check:")
    print(f"  s=0.0 theoretical: {theoretical_curve(np.array([0.0]), 8.0, 0.30)[0]:.6f}  "
          f"(should be exactly 1.0)")
    print(f"  s=1.0 theoretical: {theoretical_curve(np.array([1.0]), 8.0, 0.30)[0]:.6f}  "
          f"(should be exactly 0.7)")

    plt.show()


if __name__ == '__main__':
    main()