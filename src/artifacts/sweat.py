"""

======================
Sweat artifact processor for surface EMG.

Implements BaseArtifact for use in the simulator's artifact pipeline.

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

from src.artifacts.base import BaseArtifact


class SweatProcessor(BaseArtifact):
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
            attenuation_steepness: float = 8.0,
            powerline_harmonic_weights: tuple[float, ...] = (1.0, 0.3, 0.1, 0.03),
        ):
            # ----- Store fs-independent config -----
            self.n_channels       = n_channels
            self.powerline_freq   = float(powerline_freq)
            self.tau_sweat        = float(tau_sweat)
            self.dry_cutoff_hz    = float(dry_cutoff_hz)
            self.wet_cutoff_hz    = float(wet_cutoff_hz)
            self.max_attenuation  = float(max_attenuation)
            self.drift_scale      = float(drift_scale)
            self.rms_tau          = float(rms_tau)
            self._sigmoid_k       = float(attenuation_steepness)
            # Precomputed sigmoid values at the endpoints, used to rescale
            # the sigmoid so it hits exactly 0 at s=0 and exactly 1 at s=1.
            # This preserves the pass-through guard semantics (no attenuation
            # when sweat is off) and makes max_attenuation an accurate label
            # for the attenuation at full sweat.
            sig_at_zero            = 1.0 / (1.0 + np.exp( self._sigmoid_k * 0.5))
            sig_at_one             = 1.0 / (1.0 + np.exp(-self._sigmoid_k * 0.5))
            self._sig_at_zero      = float(sig_at_zero)
            self._sig_range        = float(sig_at_one - sig_at_zero)
            self._harmonic_weights = np.asarray(powerline_harmonic_weights, dtype=np.float64)


            # ----- Seeded RNG (independent of np.random global state) -----
            self._rng = np.random.default_rng(seed)

            # ----- Per-channel fixed factors (set once, never change) -----
            # Imbalance: uniform on the requested range. Represents physical
            # electrode impedance mismatch, independent of sweat level.
            lo, hi = imbalance_range
            self._per_channel_imbalance = self._rng.uniform(
                low=lo, high=hi, size=n_channels
            ).astype(np.float64)

            # ----- User-facing state (preserved across rebuild) -----
            # Per-channel interface degradation state, one value per channel.
            # The user sets a target vector via set_channel_sweat_levels()
            # (or set_sweat_level() for uniform mode). Current state advances
            # toward target via first-order dynamics in _update_smoothed_level().
            self._target_interface_state  = np.zeros(n_channels, dtype=np.float64)
            self._current_interface_state = np.zeros(n_channels, dtype=np.float64)

            # ----- Other runtime state (also preserved across rebuild) -----
            self._baseline_drift_value = np.zeros(n_channels, dtype=np.float64)
            self._powerline_phase = 0.0
            self._running_rms: np.ndarray | None = None
            self._lp_cache: dict = {}
            # Per-channel lowpass filter delay-line state.
            # List of length n_channels; each entry is either None (uninitialized)
            # or a numpy array of shape (filter_order,) with the current zi.
            self._lp_zi_per_channel = [None] * n_channels

            # ----- fs-dependent setup -----
            # rebuild() stores self.fs, designs the drift-noise filter, and
            # initializes filter state and cached lowpass coefficients to None
            # (the sentinel value that triggers lazy init on first process() call).
            self.rebuild(fs)
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channel_sweat_levels(self, targets) -> None:
        """
        Set per-channel target sweat levels.

        Parameters
        ----------
        targets : array-like of length n_channels
            Target sweat level for each channel. Each value clipped to [0, 1].
            The current interface state on each channel will advance toward
            its target via first-order dynamics (tau_sweat), or snap
            immediately when tau_sweat=0.
        """
        arr = np.asarray(targets, dtype=np.float64)
        if arr.shape != (self.n_channels,):
            raise ValueError(
                f"targets must have shape ({self.n_channels},), "
                f"got {arr.shape}"
            )
        self._target_interface_state = np.clip(arr, 0.0, 1.0)

    def set_sweat_level(self, target: float) -> None:
        """
        Set a uniform target sweat level across all channels.

        Convenience method for global-mode UI. Equivalent to
        set_channel_sweat_levels([target] * n_channels).

        Parameters
        ----------
        target : float
            Target sweat level, clipped to [0, 1]. Applied uniformly
            to all channels.
        """
        uniform = np.full(self.n_channels, float(target), dtype=np.float64)
        self.set_channel_sweat_levels(uniform)

    def reset(self) -> None:
        """
        Clear all runtime state. Per-channel imbalance is NOT reset
        (deterministic from the seed).
        """
        self._target_interface_state[:]  = 0.0
        self._current_interface_state[:] = 0.0
        self._lp_cache = {}
        self._lp_zi_per_channel = [None] * self.n_channels
        self._baseline_drift_value[:] = 0.0
        self._powerline_phase     = 0.0
        self._running_rms         = None
        # Drift noise filter zi is reset lazily on next use
        self._drift_noise_zi      = None

    def rebuild(self, sample_rate: float) -> None:
            """
            Recompute fs-dependent internals for a new sample rate.

            Called by the simulator when the user changes fs mid-session.
            User-facing state (slider level, running RMS, drift accumulator,
            powerline phase, per-channel constants) is preserved. Filter
            coefficients and delay-line state are discarded because they
            were designed at the old fs.
            """
            self.fs = float(sample_rate)

            # Redesign the drift-noise filter at the new rate.
            # _design_drift_noise_filter() reads self.fs.
            self._design_drift_noise_filter()

            # Invalidate cached lowpass coefficients; the next process() call
            # will redesign them in _get_lowpass_coeffs() at the new Nyquist.
            self._lp_cache = {}
            self._lp_zi_per_channel = [None] * self.n_channels
            self._drift_noise_zi = None

            # NOT touched: _target_interface_state, _current_interface_state,
            # _running_rms, _baseline_drift_value, _powerline_phase,
            # _per_channel_imbalance.

    def process(self, emg: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply sweat artifacts to one buffer of EMG.

        Parameters
        ----------
        emg : np.ndarray, shape (n_samples, n_channels)
            Input EMG samples.
        sample_rate : float
            Sampling rate in Hz. Part of the BaseArtifact contract.
            SweatProcessor stores fs internally via rebuild() and reads
            it from self.fs; this argument is accepted but not used.
            The simulator guarantees sample_rate == self.fs by calling
            rebuild() before any process() call at a new rate.

        Returns
        -------
        np.ndarray, shape (n_samples, n_channels)
            Sweat-degraded EMG. Same shape as input. Returns the original
            array (not a copy) when sweat is effectively zero.
        """
        n_samples = emg.shape[0]

        # ----- 1. Advance the smoothed sweat level toward the target -----
        self._update_smoothed_level(n_samples)
        
        # ----- 2. Bit-exact pass-through -----
        # If every channel's target and current are at/near zero, snap
        # current to 0 and return the input unchanged. Decay tail below
        # 1e-6 is invisible. If any channel has nonzero sweat, the
        # pipeline runs on all channels (we don't split the buffer).
        if (np.all(self._target_interface_state == 0.0)
                and np.all(self._current_interface_state < 0.01)):
            self._current_interface_state[:] = 0.0
            return emg
        
        # ----- 3. Effective per-channel sweat for this buffer -----
        # After the Stage 3A refactor, the per-channel state IS the
        # effective sweat vector. No global-times-modulation step needed.
        s = self._current_interface_state  # shape (C,)

        # ----- 4. Update rolling RMS so additive mechanisms can use it -----
        self._update_running_rms(emg, n_samples)

        # ----- 5. Apply mechanisms (each one is added incrementally) -----
        out = emg  # placeholder; will be reassigned by mechanisms below

        # --- Mechanism 1: amplitude attenuation (Abdoli-Eramaki et al. 2012) ---
        # Sigmoid amplitude loss vs per-channel sweat level. Real EMG
        # amplitude vs sweat shows a flat-then-drop-then-plateau shape
        # rather than a straight-line drop; the sigmoid captures that.
        # Normalized so the multiplier equals 1.0 at s=0 (preserving the
        # pass-through guard) and (1 - max_attenuation) at s=1.
        sigmoid_raw            = 1.0 / (1.0 + np.exp(-self._sigmoid_k * (s - 0.5)))   # shape (C,)
        sigmoid_normalized     = (sigmoid_raw - self._sig_at_zero) / self._sig_range  # shape (C,)
        attenuation_multiplier = 1.0 - self.max_attenuation * sigmoid_normalized      # shape (C,)
        out = emg * attenuation_multiplier                                            # shape (T, C)                   


        # --- Mechanism 2: stateful low-pass filtering (De Luca 1997; ---
        # --- Abdoli-Eramaki et al. 2012) ---
        # Cutoff slides from dry_cutoff_hz (dry, near-transparent) down to
        # wet_cutoff_hz (wet, attenuates the upper EMG band). PER-CHANNEL
        # cutoff after Stage 3A: each channel's cutoff is driven by its
        # own interface state, so clean channels see near-transparent
        # filtering while sweaty channels see aggressive filtering.
        #
        # Filter state is preserved per-channel across buffers via
        # lfilter_zi so there are no boundary discontinuities. Coefficient
        # design is cached in _lp_cache keyed by quantized cutoff, so
        # channels sharing (quantized) sweat level share designs.
        filtered = np.empty_like(out)
        for ch in range(self.n_channels):
            b_ch, a_ch = self._get_lowpass_coeffs_for_channel(s[ch])

            if self._lp_zi_per_channel[ch] is None:
                # First call: seed filter state to steady state for this
                # channel's first sample.
                zi_single = dsp.lfilter_zi(b_ch, a_ch)              # shape (4,)
                self._lp_zi_per_channel[ch] = zi_single * out[0, ch]  # shape (4,)

            filtered[:, ch], self._lp_zi_per_channel[ch] = dsp.lfilter(
                b_ch, a_ch, out[:, ch], zi=self._lp_zi_per_channel[ch]
            )

        out = filtered


        # --- Mechanism 3: baseline drift (De Luca et al. 2010) ---
        # Low-frequency wandering of the baseline voltage caused by sweat-induced
        # electrochemistry changes at the skin-electrode interface. Modeled as
        # white noise low-passed to below 1 Hz. The lowpass output is already
        # slowly-correlated and "wanders" naturally; no integrator is needed.

        # Step 1: white Gaussian noise per emg.
        noise = self._rng.standard_normal(size=(n_samples, self.n_channels))

        # Step 2: low-pass filter to sub-1-Hz content. State preserved across emgs.
        if self._drift_noise_zi is None:
            zi_single = dsp.lfilter_zi(self._drift_b, self._drift_a)  # shape (2,)
            self._drift_noise_zi = np.zeros((len(zi_single), self.n_channels))

        drift_trace, self._drift_noise_zi = dsp.lfilter(
            self._drift_b, self._drift_a, noise, axis=0, zi=self._drift_noise_zi
        )

        # Step 3: normalize drift to unit std via the filter's precomputed
        # noise gain, then scale by per-channel sweat and signal RMS.
        # Dividing by a fixed per-fs constant (not each buffer's own std)
        # makes drift both fs-independent AND continuous across streaming
        # buffers. drift_scale is now directly interpretable: the drift
        # std as a fraction of signal RMS at full sweat.
        drift_unit      = drift_trace / self._drift_noise_gain          # ~unit std
        drift_amplitude = self.drift_scale * s * self._running_rms      # shape (C,)
        out = out + drift_unit * drift_amplitude
        #
        
        
        
        # --- Mechanism 4: powerline pickup from CMRR loss (Webster 1984; Xu et al. 2020) ---
        # Sweat creates asymmetric electrode-skin impedance. Asymmetric impedance
        # breaks the differential amplifier's common-mode rejection, so powerline
        # frequency (60 Hz in NA, 50 Hz in EU) from the local wall-power
        # environment leaks into the recording as a differential signal.
        #
        # Real powerline pickup includes harmonics from grid distortion and
        # non-linear coupling at the skin-electrode interface (Mewett et al.
        # 2001). Modeled here as a sum of sinusoids at integer multiples of the
        # fundamental, with amplitude weights decaying per harmonic order.
        # Harmonics are phase-locked to the fundamental (k-th harmonic phase =
        # k * fundamental phase) because they come from the same periodic
        # distortion at the source.

        # Step 1: vectorized time indices for this buffer.
        t_indices = np.arange(n_samples)                                          # shape (T,)

        # Step 2: fundamental phase across the buffer. Continuous with previous
        # buffer via _powerline_phase.
        fundamental_phase = (
            self._powerline_phase
            + 2.0 * np.pi * self.powerline_freq * t_indices / self.fs
        )                                                                         # shape (T,)

        # Step 3: build the harmonic stack. Each harmonic k contributes
        # weight_k * sin(k * fundamental_phase). Harmonics whose frequency
        # exceeds 0.95 * Nyquist are skipped to avoid aliasing at low fs.
        nyq = 0.5 * self.fs
        max_safe_freq = 0.95 * nyq
        sine_values = np.zeros(n_samples, dtype=np.float64)                       # shape (T,)
        for k, weight in enumerate(self._harmonic_weights, start=1):
            if k * self.powerline_freq >= max_safe_freq:
                continue
            sine_values += weight * np.sin(k * fundamental_phase)

        # Step 4: update fundamental phase for next buffer. Modulo keeps it
        # bounded. Only the fundamental phase is tracked; harmonic phases
        # derive from it.
        self._powerline_phase = (
            self._powerline_phase
            + 2.0 * np.pi * self.powerline_freq * n_samples / self.fs
        ) % (2.0 * np.pi)

        # Step 5: per-channel amplitude. Imbalance is fixed per channel; sweat
        # and RMS scale dynamically. Shape (C,).
        powerline_amplitude = (
            self._per_channel_imbalance * s * self._running_rms
        )

        # Step 6: outer-product broadcast: (T,) x (C,) -> (T, C).
        powerline_signal = sine_values[:, None] * powerline_amplitude[None, :]    # shape (T, C)

        out = out + powerline_signal

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_smoothed_level(self, n_samples: int) -> None:
        """
        Advance per-channel interface state toward per-channel target
        via first-order dynamics. Asymmetric tau: slow buildup
        (tau_sweat, default 5s), fast decay (tau_sweat * 0.2, default 1s).

        Each channel is updated independently. Channels whose target
        is 0 use the fast-decay tau; channels whose target is nonzero
        use the slow-buildup tau. This means dropping the sweat on one
        channel to zero decays it quickly on that channel alone.
        """
        # Offline mode: snap immediately. Must come BEFORE tau math,
        # otherwise tau_sweat=0 propagates into a divide-by-zero.
        if self.tau_sweat <= 0.0:
            self._current_interface_state[:] = self._target_interface_state
            return

        # Per-channel effective tau vector.
        # Channels with target == 0 get the fast-decay tau; others get slow.
        slow_tau  = self.tau_sweat
        fast_tau  = self.tau_sweat * 0.2
        tau_vec   = np.where(
            self._target_interface_state == 0.0,
            fast_tau,
            slow_tau,
        )

        dt        = n_samples / self.fs
        alpha_vec = 1.0 - np.exp(-dt / tau_vec)          # shape (C,)
        self._current_interface_state += alpha_vec * (
            self._target_interface_state - self._current_interface_state
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
        """
        Design the sub-1-Hz lowpass that turns white noise into slow
        baseline drift, and precompute the filter's noise gain.

        Noise gain is the std of the filter output for unit-variance
        white-noise input, computed from the impulse response. We divide
        the drift trace by this constant so drift amplitude is set by
        drift_scale alone, independent of fs. The raw filter output std
        depends on fs (the fixed 0.5 Hz cutoff sits at a different
        fraction of Nyquist at each rate), which is what made the old
        fixed * 30.0 scaling explode at high sample rates. Using a
        precomputed constant (not each buffer's own std) also keeps the
        drift continuous across streaming buffers.
        """
        nyq    = 0.5 * self.fs
        cutoff = min(0.5, 0.45 * nyq)               # clamp for tiny fs
        self._drift_b, self._drift_a = dsp.butter(
            N=2, Wn=cutoff / nyq, btype='low'
        )

        # Noise gain = sqrt(sum of squared impulse response) = output std
        # for unit-variance white-noise input. Long impulse for a stable
        # estimate (the 0.5 Hz filter has a long tail).
        imp = np.zeros(16384)
        imp[0] = 1.0
        h = dsp.lfilter(self._drift_b, self._drift_a, imp)
        self._drift_noise_gain = float(np.sqrt(np.sum(h ** 2)))
        if self._drift_noise_gain < 1e-12:
            self._drift_noise_gain = 1.0
    
    def _get_lowpass_coeffs_for_channel(self, s_channel: float):
        """
        Return cached Butterworth low-pass coefficients (b, a) for a
        single channel's current interface state. Cutoff slides linearly
        from dry_cutoff_hz at s=0 down to wet_cutoff_hz at s=1.

        Quantized to nearest 5 Hz. Coefficient design is cached in
        self._lp_cache keyed by quantized cutoff, so channels with the
        same (quantized) sweat level share designs. In global mode all
        8 channels hit the same cache entry.
        """
        # Step 1: linear interpolation between dry and wet cutoffs for this channel.
        cutoff_hz = (
            self.dry_cutoff_hz
            + (self.wet_cutoff_hz - self.dry_cutoff_hz) * float(s_channel)
        )

        # Step 2: quantize to nearest 5 Hz for caching.
        quantized = round(cutoff_hz / 5.0) * 5.0

        # Step 3: clamp cutoff safely below Nyquist and above 1 Hz.
        nyq = 0.5 * self.fs
        max_safe_cutoff = 0.95 * nyq
        min_safe_cutoff = 1.0
        quantized = max(min_safe_cutoff, min(quantized, max_safe_cutoff))

        # Step 4: cache lookup.
        if quantized in self._lp_cache:
            return self._lp_cache[quantized]

        b, a = dsp.butter(N=4, Wn=quantized / nyq, btype='low')
        self._lp_cache[quantized] = (b, a)
        return b, a