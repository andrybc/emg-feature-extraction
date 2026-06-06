"""

======================
Sweat artifact processor for surface EMG.

Models four physical mechanisms by which sweat degrades sEMG recordings:
    1. Amplitude attenuation       (Abdoli-Eramaki et al. 2012)
    2. Low-pass filtering          (De Luca 1997; Abdoli-Eramaki et al. 2012)
    3. Baseline drift              (De Luca et al. 2010)
    4. CMRR loss from impedance    (Webster 1984; Xu et al. 2020)
       imbalance -> powerline pickup

Designed for two use cases from one class:
    - Real-time streaming inside the EMG simulator (kHz sample rates,
      small buffers, state preserved across calls)
    - Offline batch processing on recorded NumPy arrays (single shot,
      no temporal smoothing)

The difference is one parameter: tau_sweat. Streaming uses ~5 seconds.
Offline uses 0.
"""

import numpy as np
import scipy.signal as dsp


class SweatProcessor:
    """
    Adds physically-motivated sweat artifacts to multi-channel EMG.

    Parameters
    ----------
    n_channels : int
        Number of EMG channels in each buffer.
    fs : float
        Sample rate in Hz. Used for filter design and time-constant math.
    powerline_freq : float, default 60.0
        Local powerline frequency. 60 Hz in North America, 50 Hz in EU.
    tau_sweat : float, default 5.0
        Time constant (seconds) for the slider-to-actual sweat level
        first-order response. Set to 0 for offline processing.
    seed : int or None, default None
        Seed for per-channel randomness. Pass an int for reproducibility.
    dry_cutoff_hz : float, default 400.0
        Low-pass cutoff at sweat_level = 0. Effectively no filtering.
    wet_cutoff_hz : float, default 150.0
        Low-pass cutoff at sweat_level = 1. Matches mean-frequency drop
        reported in Abdoli-Eramaki et al. 2012.
    max_attenuation : float, default 0.30
        Fractional amplitude loss at sweat_level = 1.
    drift_scale : float, default 0.5
        Baseline drift amplitude as a fraction of signal RMS at sweat = 1.
    imbalance_range : (float, float), default (0.0, 0.3)
        Per-channel electrode impedance imbalance, drawn uniformly. Drives
        the 60 Hz pickup amplitude.
    rms_tau : float, default 0.2
        Time constant (seconds) for the rolling RMS estimator. RMS is used
        to scale the additive drift and powerline mechanisms.
    """

    def __init__(
        self,
        n_channels: int,
        fs: float,
        powerline_freq: float = 60.0,
        tau_sweat: float = 5.0,
        seed: int | None = None,
        dry_cutoff_hz: float = 400.0,
        wet_cutoff_hz: float = 150.0,
        max_attenuation: float = 0.30,
        drift_scale: float = 0.5,
        imbalance_range: tuple[float, float] = (0.0, 0.3),
        rms_tau: float = 0.2,
    ):
        # ----- Store config -----
        self.n_channels       = n_channels
        self.fs               = float(fs)
        self.powerline_freq   = float(powerline_freq)
        self.tau_sweat        = float(tau_sweat)
        self.dry_cutoff_hz    = float(dry_cutoff_hz)
        self.wet_cutoff_hz    = float(wet_cutoff_hz)
        self.max_attenuation  = float(max_attenuation)
        self.drift_scale      = float(drift_scale)
        self.rms_tau          = float(rms_tau)

        # ----- Seeded RNG (independent of np.random global state) -----
        self._rng = np.random.default_rng(seed)

        # ----- Per-channel fixed factors (set once, never change) -----
        # Modulation: normal(1.0, 0.15) clipped to [0.6, 1.4]
        raw_mod = self._rng.normal(loc=1.0, scale=0.15, size=n_channels)
        self._per_channel_modulation = np.clip(raw_mod, 0.6, 1.4).astype(np.float64)

        # Imbalance: uniform on the requested range
        lo, hi = imbalance_range
        self._per_channel_imbalance = self._rng.uniform(
            low=lo, high=hi, size=n_channels
        ).astype(np.float64)

        # ----- Target and smoothed sweat level -----
        self._target_sweat_level  = 0.0
        self._current_sweat_level = 0.0

        # ----- Mutable runtime state (populated by reset()) -----
        # Filter delay-line state for the variable-cutoff low-pass.
        # We'll allocate it the first time process() runs, because the
        # cutoff (and therefore filter coefficients) depend on sweat level.
        # For now: placeholder.
        self._lp_zi: np.ndarray | None = None
        self._lp_last_cutoff: float | None = None  # for detecting cutoff change
        self._lp_b: np.ndarray | None = None       # cached numerator coefficients
        self._lp_a: np.ndarray | None = None       # cached denominator coefficients

        # Filter state for the noise that drives baseline drift.
        # Cutoff is fixed (~0.5 Hz), so we can design this once.
        self._drift_noise_zi: np.ndarray | None = None
        self._design_drift_noise_filter()

        # Running cumulative-sum value per channel for baseline drift.
        self._baseline_drift_value = np.zeros(n_channels, dtype=np.float64)

        # Powerline sine phase (radians). Continuous across buffers.
        self._powerline_phase = 0.0

        # Rolling RMS per channel. None means "uninitialized".
        # On first buffer, we'll seed it with the actual measurement
        # instead of starting from zero (which would cause a ramp).
        self._running_rms: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_sweat_level(self, target: float) -> None:
        """Set the target sweat level. Will be clipped to [0, 1]."""
        self._target_sweat_level = float(np.clip(target, 0.0, 1.0))

    def reset(self) -> None:
        """
        Clear all runtime state. Per-channel modulation and imbalance
        are NOT reset (those are deterministic from the seed).
        """
        self._target_sweat_level  = 0.0
        self._current_sweat_level = 0.0
        self._lp_zi               = None
        self._lp_last_cutoff      = None
        self._lp_b                = None
        self._lp_a                = None
        self._baseline_drift_value[:] = 0.0
        self._powerline_phase     = 0.0
        self._running_rms         = None
        # Drift noise filter zi is reset lazily on next use
        self._drift_noise_zi      = None

    def process(self, buffer: np.ndarray) -> np.ndarray:
        """
        Apply sweat artifacts to one buffer of EMG.

        Parameters
        ----------
        buffer : np.ndarray, shape (n_samples, n_channels)
            Input EMG samples.

        Returns
        -------
        np.ndarray, shape (n_samples, n_channels)
            Sweat-degraded EMG. Same shape as input. Returns the original
            array (not a copy) when sweat is effectively zero.
        """
        n_samples = buffer.shape[0]

        # ----- 1. Advance the smoothed sweat level toward the target -----
        self._update_smoothed_level(n_samples)

        # ----- 2. Bit-exact pass-through (Q4 option A) -----
        # If both target and current are at/near zero, snap current to 0
        # and return the input unchanged. Decay tail below 1e-6 is invisible.
        if self._target_sweat_level == 0.0 and self._current_sweat_level < 0.01:
            self._current_sweat_level = 0.0
            return buffer

        # ----- 3. Effective per-channel sweat for this buffer -----
        s = self._current_sweat_level * self._per_channel_modulation  # (C,)

        # ----- 4. Update rolling RMS so additive mechanisms can use it -----
        self._update_running_rms(buffer, n_samples)

        # ----- 5. Apply mechanisms (each one is added incrementally) -----
        out = buffer  # placeholder; will be reassigned by mechanisms below

        # --- Mechanism 1: amplitude attenuation (Abdoli-Eramaki et al. 2012) ---
        # Linear amplitude loss proportional to per-channel sweat level. At
        # sweat=1 each channel drops by max_attenuation (default 30%). The
        # multiplier has shape (C,) and broadcasts across the (T, C) buffer.
        attenuation_multiplier = 1.0 - self.max_attenuation * s         # shape (C,)
        out = buffer * attenuation_multiplier                           # shape (T, C)


        # --- Mechanism 2: stateful low-pass filtering (De Luca 1997; ---
        # --- Abdoli-Eramaki et al. 2012) ---
        # Cutoff slides from 400 Hz (dry, near-transparent) down to 150 Hz
        # (wet, attenuates the upper EMG band). Filter state is preserved
        # across buffers via lfilter_zi so there are no boundary discontinuities.
        b, a = self._get_lowpass_coeffs()

        if self._lp_zi is None:
            # First call after construction or reset: seed the filter state
            # using lfilter_zi so the filter starts in a steady state for
            # the current input level. Avoids a startup transient.
            # zi shape needs to be (filter_order, n_channels) for axis=0 filtering.
            zi_single = dsp.lfilter_zi(b, a)                      # shape (4,)
            # Scale each channel's initial state by that channel's first sample.
            # buffer[0] has shape (C,); zi_single[:, None] has shape (4, 1).
            self._lp_zi = zi_single[:, None] * buffer[0][None, :]  # shape (4, C)

        out, self._lp_zi = dsp.lfilter(b, a, out, axis=0, zi=self._lp_zi)


        # --- Mechanism 3: baseline drift (De Luca et al. 2010) ---
        # Low-frequency wandering of the baseline voltage caused by sweat-induced
        # electrochemistry changes at the skin-electrode interface. Modeled as
        # white noise low-passed to below 1 Hz. The lowpass output is already
        # slowly-correlated and "wanders" naturally; no integrator is needed.

        # Step 1: white Gaussian noise per buffer.
        noise = self._rng.standard_normal(size=(n_samples, self.n_channels))

        # Step 2: low-pass filter to sub-1-Hz content. State preserved across buffers.
        if self._drift_noise_zi is None:
            zi_single = dsp.lfilter_zi(self._drift_b, self._drift_a)  # shape (2,)
            self._drift_noise_zi = np.zeros((len(zi_single), self.n_channels))

        drift_trace, self._drift_noise_zi = dsp.lfilter(
            self._drift_b, self._drift_a, noise, axis=0, zi=self._drift_noise_zi
        )

        # Step 3: scale by per-channel sweat and signal RMS, then add to signal.
        # drift_trace already has std around 0.03 from the lowpass alone. The
        # scale factor needs to bring this up to something visible relative to
        # signal RMS. We boost by a fixed factor that makes drift comparable to
        # (not larger than) signal amplitude at full sweat.
        drift_amplitude = self.drift_scale * s * self._running_rms * 30.0  # shape (C,)
        out = out + drift_trace * drift_amplitude
        #
        
        
        
        # --- Mechanism 4: powerline pickup from CMRR loss (Webster 1984; Xu et al. 2020) ---
        # Sweat creates asymmetric electrode-skin impedance. Asymmetric impedance
        # breaks the differential amplifier's common-mode rejection, so 60 Hz
        # (or 50 Hz in Europe) from the local wall-power environment leaks into
        # the recording as a differential signal. Amplitude per channel scales
        # with that channel's fixed imbalance factor, the current sweat level,
        # and the rolling signal RMS.

        # Step 1: vectorized sine values across the buffer.
        # Phase advances continuously across buffers via _powerline_phase.
        t_indices = np.arange(n_samples)                                          # shape (T,)
        phase_per_sample = (
            self._powerline_phase
            + 2.0 * np.pi * self.powerline_freq * t_indices / self.fs
        )
        sine_values = np.sin(phase_per_sample)                                    # shape (T,)

        # Step 2: update phase for next buffer call. Modulo keeps it bounded.
        self._powerline_phase = (
            self._powerline_phase
            + 2.0 * np.pi * self.powerline_freq * n_samples / self.fs
        ) % (2.0 * np.pi)

        # Step 3: per-channel amplitude. Imbalance is fixed per channel; sweat
        # and RMS scale dynamically. Shape (C,).
        powerline_amplitude = (
            self._per_channel_imbalance * s * self._running_rms
        )

        # Step 4: outer-product-style broadcast: (T,) × (C,) -> (T, C)
        # sine_values[:, None] has shape (T, 1); powerline_amplitude[None, :] has shape (1, C).
        # The product is (T, C), where every row is the same sine value times the
        # per-channel amplitude vector.
        powerline_signal = sine_values[:, None] * powerline_amplitude[None, :]    # shape (T, C)

        out = out + powerline_signal        
        
        

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

        def _update_smoothed_level(self, n_samples: int) -> None:
            """
            Move _current_sweat_level toward _target_sweat_level using a
            first-order step. dt is the duration of this buffer.
            """
            # Offline mode: snap immediately. Must come BEFORE effective_tau
            # is computed, otherwise tau_sweat=0 propagates into a divide-by-zero.
            if self.tau_sweat <= 0.0:
                self._current_sweat_level = self._target_sweat_level
                return

            # Use a faster time constant when decaying toward zero. This makes
            # the "Dry" button feel responsive without sacrificing the slow buildup.
            if self._target_sweat_level == 0.0:
                effective_tau = self.tau_sweat * 0.2   # 1-second decay
            else:
                effective_tau = self.tau_sweat

            dt    = n_samples / self.fs
            alpha = 1.0 - np.exp(-dt / effective_tau)
            self._current_sweat_level += alpha * (
                self._target_sweat_level - self._current_sweat_level
            )
        
    def _update_running_rms(self, buffer: np.ndarray, n_samples: int) -> None:
        """
        Exponentially smoothed RMS per channel. Seeded with the first
        buffer's value to avoid a startup ramp.
        """
        # Instantaneous mean-square for this buffer, per channel.
        inst_ms = np.mean(buffer ** 2, axis=0)  # shape (C,)

        if self._running_rms is None:
            # First call: seed with the true value, not zero.
            self._running_rms = np.sqrt(inst_ms)
            return

        dt    = n_samples / self.fs
        alpha = 1.0 - np.exp(-dt / self.rms_tau)
        # Smooth the mean-square, then sqrt to get RMS. Smoothing MS is
        # mathematically cleaner than smoothing RMS directly.
        ms_smoothed       = self._running_rms ** 2
        ms_smoothed       = ms_smoothed + alpha * (inst_ms - ms_smoothed)
        self._running_rms = np.sqrt(ms_smoothed)

    def _design_drift_noise_filter(self) -> None:
        """..."""
        nyq    = 0.5 * self.fs
        cutoff = min(0.5, 0.45 * nyq)               # clamp for tiny fs
        self._drift_b, self._drift_a = dsp.butter(
            N=2, Wn=cutoff / nyq, btype='low'
        )
    
    def _get_lowpass_coeffs(self):
        """
        Return cached Butterworth low-pass coefficients (b, a), redesigning
        only when the quantized cutoff changes. Cutoff slides linearly from
        dry_cutoff_hz at sweat=0 down to wet_cutoff_hz at sweat=1.
        """
        # Step 1: linear interpolation between dry and wet cutoffs.
        s_scalar = self._current_sweat_level
        cutoff_hz = (
            self.dry_cutoff_hz
            + (self.wet_cutoff_hz - self.dry_cutoff_hz) * s_scalar
        )

        # Step 2: quantize to nearest 5 Hz to enable design caching.
        quantized = round(cutoff_hz / 5.0) * 5.0

        # Step 3: clamp cutoff to be safely below Nyquist. At very low sample
        # rates (e.g. 250 Hz, Nyquist 125 Hz), the dry cutoff of 400 Hz would
        # exceed Nyquist and crash butter(). Clamp to 0.45 * Nyquist as a safe
        # upper bound; 0.45 gives some margin from the hard Nyquist limit.
        nyq = 0.5 * self.fs
        max_safe_cutoff = 0.45 * nyq                    # 45% of fs/2
        min_safe_cutoff = 1.0                           # 1 Hz floor
        quantized = max(min_safe_cutoff, min(quantized, max_safe_cutoff))

        # Step 4: rebuild only on quantized-cutoff change.
        if quantized != self._lp_last_cutoff:
            self._lp_b, self._lp_a = dsp.butter(
                N=4, Wn=quantized / nyq, btype='low'
            )
            self._lp_last_cutoff = quantized

        return self._lp_b, self._lp_a