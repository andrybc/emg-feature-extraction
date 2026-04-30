"""Utilities for generating synthetic EMG-like signals for learning and testing."""

import numpy as np


def generate_fake_emg(
    duration_s: float = 5.0,
    sampling_rate_hz: int = 500,
    n_channels: int = 8,
    contraction_start_s: float = 1.5,
    contraction_end_s: float = 3.5,
    baseline_noise_std: float = 0.03,
    contraction_noise_std: float = 0.20,
    random_seed: int = 42,
):
    """Generate a simple fake multi-channel EMG-like signal.

    Returns:
        time: shape (samples,)
        emg: shape (samples, channels)
        envelope: shape (samples,)
    """
    rng = np.random.default_rng(random_seed)
    n_samples = int(duration_s * sampling_rate_hz)
    time = np.arange(n_samples) / sampling_rate_hz

    active = (time >= contraction_start_s) & (time <= contraction_end_s)

    envelope = np.full(n_samples, baseline_noise_std)
    envelope[active] = contraction_noise_std

    emg = rng.normal(loc=0.0, scale=envelope[:, None], size=(n_samples, n_channels))

    # Add small channel-specific differences.
    channel_gains = rng.uniform(0.7, 1.3, size=n_channels)
    emg = emg * channel_gains

    return time, emg, envelope
