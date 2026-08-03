"""
test_generator_spectrum.py
==========================
Confirm the EMG signal generator's output spectrum occupies the same
absolute-Hz band regardless of sample rate. Before the fix, the
hardcoded shaping filter scaled with Nyquist, leaking energy above
400 Hz at high fs. After the fix, the energy should stay in ~20-350 Hz
at every rate.
"""

import numpy as np

# Import the generator from the simulator module.
import importlib.util
spec = importlib.util.spec_from_file_location("emg_simulator", "emg_simulator.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
EMGSignalGenerator = mod.EMGSignalGenerator


def spectrum_stats(fs):
    gen = EMGSignalGenerator(fs=fs)
    gen.set_gesture('Hand Closed')
    # Generate 4 seconds.
    n = int(fs * 4)
    samples, _ = gen.generate_samples(n_samples=n)
    x = samples[:, 0]
    x = x - x.mean()

    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    mag   = np.abs(np.fft.rfft(x))
    power = mag ** 2

    # Fraction of energy below 400 Hz and above 400 Hz.
    below = power[freqs < 400].sum()
    above = power[freqs >= 400].sum()
    total = power.sum()

    # Spectral centroid (energy-weighted mean frequency).
    centroid = (freqs * power).sum() / total
    return below / total, above / total, centroid


print(f"{'fs':>6}  {'%below400':>10}  {'%above400':>10}  {'centroid Hz':>12}")
for fs in [1000.0, 2000.0, 4000.0]:
    b, a, c = spectrum_stats(fs)
    print(f"{fs:>6.0f}  {100*b:>10.1f}  {100*a:>10.1f}  {c:>12.1f}")