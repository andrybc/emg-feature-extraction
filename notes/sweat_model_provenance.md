# Sweat Model Provenance

Status of every design decision in `src/artifacts/sweat.py`, sorted into
literature-grounded behavior versus modeling choices. Written for the manuscript:
when we cite a paper, this document records the exact claim we are relying on and
whether it is a measurement or an assertion. When a value is a modeling choice, this
document says so plainly so we never over-claim empirical support we do not have.

Legend for the Status column:

- **Grounded (measured)**: a cited paper reports an experimental result that directly supports this.
- **Grounded (asserted)**: a cited paper states this as a physical argument, but does not measure it.
- **Adjacent**: a paper motivates the general idea, but our specific implementation or value goes beyond what it reports.
- **Modeling choice**: chosen by us for plausibility or engineering reasons; no citation backs the specific value.

Verified sources are papers we have read directly. Unverified sources are cited in the
code but not yet read against the original PDF; each is flagged as an action item before
manuscript submission.

---

## Source inventory

**Verified (read directly):**

- Abdoli-Eramaki, M., Damecour, C., Christenson, J., & Stevenson, J. (2012). The effect of perspiration on the sEMG amplitude and power spectrum. *Journal of Electromyography and Kinesiology*, 22(6), 908-913.
- Ray, G.C., & Guha, S.K. (1983). Equivalent Electrical Representation of the Sweat Layer and Gain Compensation of the EMG Amplifier. *IEEE Transactions on Biomedical Engineering*, BME-30(2), 130-132.
- De Luca, C.J. (1997). The Use of Surface Electromyography in Biomechanics. *Journal of Applied Biomechanics*, 13(2), 135-163.
- Kyranou, I., Vijayakumar, S., & Erden, M.S. (2018). Causes of Performance Degradation in Non-invasive Electromyographic Pattern Recognition in Upper Limb Prostheses. *Frontiers in Neurorobotics*, 12, 58.

**Unverified (cited in code, not yet read against the source PDF):**

- De Luca et al. 2010 (baseline drift) — action item: pull and verify.
- Webster 1984 (powerline CMRR) — action item: pull and verify.
- Xu et al. 2020 (dry-electrode CMRR) — action item: pull and verify.
- Mewett et al. 2001 (powerline harmonics) — action item: pull and verify.

---

## Mechanism 1: Amplitude attenuation

**What the code does:** multiplies the signal by a per-channel factor that decreases as
sweat rises, using a normalized sigmoid centered at sweat = 0.5.

| Element | Value | Status | Evidence |
|---|---|---|---|
| Sweat reduces sEMG amplitude | — | Grounded (measured) | Two independent papers, see below |
| `max_attenuation` | 0.30 | Adjacent | In range of measured losses, not a fitted value |
| Sigmoid shape (vs linear) | k = 8 | Modeling choice | Papers show roughly linear; sigmoid is our bounded-saturation choice |

**Evidence, Abdoli-Eramaki 2012:** The paper measured amplitude loss directly. Abstract:
sweat accumulation "dampens the amplitude of the EMG signal in a predictable way (r = .88
and .97 for double and single snap electrodes, respectively) with almost 2% and 3%
deterioration for every 0.02 mm of sweat." Figure 3 shows amplitude dropping to roughly
52% of the dry condition at 0.2 mm sweat for double-snap electrodes.

**Evidence, Ray & Guha 1983:** An independent, older measurement with a physical model. The
sweat layer acts as a shunt resistance across the electrode pair, partially short-circuiting
the signal. Their Figure 1 shows signal decreases of 13%, 33%, and 44% at 28, 32, and 35 C
as sweat accumulates over ~100 minutes. Abdoli-Eramaki explicitly notes their own results
are consistent with Ray & Guha.

**Why the sigmoid is a modeling choice, not a fit:** Abdoli-Eramaki's Figure 3 is close to
linear over their measured range (0 to 0.2 mm), and the paper reports a linear regression
(the r = .88 / .97 figures are for a linear fit). We use a saturating sigmoid instead of a
strict line for two reasons: (1) a linear model extrapolated past sweat = 1 would predict
negative amplitude, which is unphysical; (2) impedance degradation is a saturating process
with a floor (dry contact) and ceiling (fully wet). The sigmoid matches the linear data
closely across the middle range and adds physically motivated saturation at the tails. The
steepness k is exposed as `--sweat-steepness` (default 8, k = 2 recovers near-linear
behavior), so a reviewer can dial it back toward the paper's linear form. **This is
documented as a modeling choice, not an empirical fit.**

---

## Mechanism 2: Lowpass filtering

**What the code does:** applies a per-channel Butterworth lowpass whose cutoff slides from
400 Hz (dry) down to 150 Hz (fully wet).

| Element | Value | Status | Evidence |
|---|---|---|---|
| Sweat filters high frequencies | — | Grounded (asserted), contradicted by one measurement | De Luca 1997 asserts; Abdoli-Eramaki 2012 tested and found none |
| Dry cutoff | 400 Hz | Modeling choice | Sits inside De Luca's recommended 20-500 Hz band |
| Wet cutoff | 150 Hz | Modeling choice | Not a measured value |
| Linear cutoff slide | — | Modeling choice | — |

**This is the least empirically supported of the four mechanisms, and the code has been
written so it can be commented out cleanly if we decide the evidence does not justify it.**
See "Removing Mechanism 2" below.

**Evidence for, De Luca 1997 (p. 142):** In the discussion of electrode detection-surface
spacing, De Luca writes that the surfaces "may be shunted electrically if the surface of the
skin becomes moist with sweat, which is conductive. The electrical shunting decreases the
signal amplitude, deteriorates the signal to noise ratio, and may filter out the higher
frequency components." This is the original source of the lowpass idea. Two caveats we must
carry into the manuscript: (1) it is a hedged physical argument ("may filter"), not a
measurement, and there is no figure, cutoff, or quantified effect; (2) even in this sentence
the confident claims are about amplitude and SNR, with the spectral effect listed last and
qualified.

**Evidence against, Abdoli-Eramaki 2012:** This is the one paper in our set that
experimentally tested for a spectral effect of sweat, and it found none. Abstract: "MF was
not influenced by sweat accumulation even under the extreme wet condition." Section 3.2:
"sweat layer was not an important factor for MF." Conclusion: "The influence of sweat
accumulation was shown to influence the amplitude of the sEMG but not the MF." Figure 4
shows mean frequency staying flat (around 100% of the dry condition) across all sweat
thicknesses.

**Honest position for the manuscript:** the literature is split. A physical rationale (De
Luca 1997) supports a sweat-induced lowpass, but the only direct experimental test in our
source set (Abdoli-Eramaki 2012) did not detect a spectral change. When an assertion and a
measurement conflict, the measurement generally carries more weight, particularly here
because Abdoli-Eramaki designed their study specifically to test it. We include the mechanism
because the physical shunting argument is sound and De Luca states it, but we label it as
speculative and keep it removable.

**On the cutoff values:** 400 Hz and 150 Hz are the filter knee frequencies, a different
quantity from the mean frequency Abdoli-Eramaki measured. Neither paper reports these values.
They are chosen so the dry cutoff is near-transparent for real EMG (whose energy sits below
~400 Hz; De Luca 1997 recommends a 20-500 Hz recording band, p. 143) and the wet cutoff
removes a visible portion of the upper band. Both are modeling choices.

### Removing Mechanism 2

If we decide the evidence does not justify a spectral effect, Mechanism 2 can be disabled
without touching the other three mechanisms:

- Setting `wet_cutoff_hz` equal to `dry_cutoff_hz` makes the cutoff constant at all sweat
  levels, so the lowpass never changes and the mechanism is effectively inert (the filter
  still runs but does nothing sweat-dependent). This is the least invasive "off switch" and
  needs no code change, only a constructor argument.
- To remove it entirely, comment out the Mechanism 2 block in `process()` (the per-channel
  lowpass loop) and the `_get_lowpass_coeffs_for_channel` call. The `BaseArtifact` contract,
  the pass-through guard, and Mechanisms 1, 3, and 4 are unaffected because they do not depend
  on the lowpass output.

Recommendation: keep Mechanism 2 for now, labeled speculative, and revisit once we have real
sweaty recordings (see "Open validation items"). If real data shows no spectral shift,
consistent with Abdoli-Eramaki, disable it via the `wet_cutoff_hz = dry_cutoff_hz` route.

---

## Mechanism 3: Baseline drift

**What the code does:** generates white noise, lowpasses it below ~0.5 Hz to produce slow
baseline wander, normalizes it by the filter's precomputed noise gain, and scales it by
`drift_scale`, sweat level, and running signal RMS.

| Element | Value | Status | Evidence |
|---|---|---|---|
| Sweat causes sub-1-Hz baseline drift | — | Grounded (asserted), source unverified | De Luca et al. 2010, cited in code, not yet read |
| `drift_scale` (drift std / signal RMS at full sweat) | 0.5 | Modeling choice | Magnitude eyeballed on the simulator display; not validated against real recordings |
| Noise-gain normalization | — | Engineering correctness | Makes drift fs-invariant; not a modeling claim |

**Physical rationale:** the skin-electrode interface is a small electrochemical cell. Sweat
changes the ionic concentration at the contact, shifting the half-cell potential, which
appears at the amplifier input as a slowly wandering DC offset. This is standard biopotential
electrode theory and is consistent with De Luca 1997's recommendation (p. 158) that the EMG
signal be recorded with a 20 Hz low-frequency cutoff specifically to reject baseline and
motion-artifact content, which implies such low-frequency content exists and must be removed.

**Honest limitation on magnitude:** the drift *amplitude* (`drift_scale`) is not grounded in
any measurement. It was tuned by eye on the simulator's own display so the baseline wanders
visibly without overwhelming the signal. It is not compared to any real sweaty recording.
This is a modeling choice and is the clearest candidate for calibration once real data is
available (see "Open validation items"). The De Luca et al. 2010 citation for the drift
*form* (sub-1-Hz wander) is in the code but has not yet been read against the source PDF and
is flagged as an action item.

---

## Mechanism 4: Powerline pickup from CMRR loss

**What the code does:** adds a harmonic stack at 60/120/180/240 Hz whose amplitude scales
with a per-channel electrode-imbalance factor, sweat level, and running RMS.

| Element | Value | Status | Evidence |
|---|---|---|---|
| Sweat degrades CMRR, letting powerline in | — | Grounded (asserted), sources unverified | Webster 1984, Xu 2020, cited in code, not yet read |
| Harmonic stack exists | 60/120/180/240 Hz | Adjacent, source unverified | Mewett 2001, cited in code, not yet read |
| Harmonic weights | 1.0, 0.3, 0.1, 0.03 | Modeling choice | Geometric decay chosen for plausibility, not fitted |
| Amplitude scales with RMS | — | Modeling choice | — |

**Physical rationale:** a differential amplifier rejects powerline interference only when the
two electrodes present matched impedance. Sweat changes the skin-electrode impedance
asymmetrically, breaking the match, degrading the common-mode rejection ratio, and letting
the ambient 60 Hz field (and its harmonics) couple into the differential signal. This is
standard instrumentation theory. Ray & Guha 1983 (verified) supports the adjacent premise
that sweat changes the effective electrode impedance, which is the input to the CMRR argument,
though Ray & Guha discuss amplitude shunting rather than powerline pickup specifically.

**Honest limitations:** the CMRR-degradation citations (Webster 1984, Xu 2020) and the
harmonic-content citation (Mewett 2001) are in the code but have not been read against the
source PDFs; all three are action items. The harmonic weights (1.0, 0.3, 0.1, 0.03) are a
geometric decay chosen for plausibility, not fitted to any measured spectrum. The
amplitude-scales-with-RMS behavior is a modeling choice, not a physical law.

---

## Item 4: The 240 Hz harmonic and the Nyquist safety margin

The powerline mechanism skips any harmonic whose frequency reaches 0.95 times the Nyquist
frequency (0.5 times the sample rate). This is a signal-processing safety decision, not a
modeling claim.

A sinusoid at frequency f is represented without aliasing only if f is below Nyquist. Very
close to Nyquist, a digital sinusoid is poorly represented (few samples per cycle) and can
produce artifacts. The 0.95 factor leaves a 5% margin below Nyquist.

Consequences at common rates:

- **1000 Hz** (Nyquist 500 Hz): ceiling is 475 Hz. The 240 Hz 4th harmonic is well below and
  is included. All four harmonics present.
- **500 Hz** (Nyquist 250 Hz): ceiling is 237.5 Hz. The 240 Hz 4th harmonic is above the
  ceiling and is skipped, because at 500 Hz sampling, 240 Hz is 96% of Nyquist and would be
  poorly represented. The fundamental and 2nd/3rd harmonics (60/120/180 Hz) remain.

So excluding 240 Hz at 500 Hz is a deliberate anti-aliasing decision, not a modeling choice
about how sweat behaves. The same 0.95 margin is used for the shaping filter in the generator.

---

## Item 5: Which sample rates are scientifically supported

The GUI offers several sample rates. They are not all equally appropriate for EMG research.

Real surface EMG energy sits roughly in the 20-450 Hz band. De Luca 1997 (p. 143,
Recommendation 1) recommends a recording bandwidth of "20-500 Hz with a roll-off at least
12 dB/octave." By the Nyquist criterion, capturing content up to ~450-500 Hz requires a
sample rate of at least ~1000 Hz.

| Sample rate | Nyquist | Status | Note |
|---|---|---|---|
| Below 1000 Hz | Below 500 Hz | Not recommended | Undersamples the real EMG band; upper-band content is lost or aliased |
| 1000 Hz | 500 Hz | Supported (recommended default) | Matches the De Luca 20-500 Hz band with margin; standard for sEMG |
| 2000 Hz | 1000 Hz | Supported (oversampled) | Valid; captures the full EMG band with headroom. No new signal above ~500 Hz, just more samples |
| 4000 Hz and above | 2000 Hz+ | Available (heavily oversampled) | Valid but no scientific benefit for EMG; larger files, more compute. Useful only for specific high-rate experiments |

Guidance: **1000 Hz is the recommended default for EMG-representative work.** Higher rates are
scientifically valid (the fixes in the generator and drift mechanisms make the signal and
artifacts spectrally invariant across rates) but oversample the EMG band. Rates below 1000 Hz
are operationally available but not scientifically recommended, because they undersample real
EMG content.

---

## Item 2: Tolerance for the sample-rate invariance claim

"Sample-rate invariant" means the model's behavior does not change materially when the sample
rate changes. It does not mean bit-identical output across rates, which is not achievable
because any realizable filter has a finite skirt and higher Nyquist gives that skirt more
spectral room.

Measured invariance after the generator and drift fixes (band-limited, EMG-like input):

- **Generator spectrum:** spectral centroid holds within roughly 186-198 Hz across
  fs = 1000/2000/4000, and energy above 400 Hz stays under ~3.5% at all rates.
- **1% sweat amplitude:** amplitude ratio stays at or above ~0.96 across all three rates
  (the original high-fs symptom is gone).
- **100% sweat amplitude:** ratio holds within roughly 0.72-0.65 across the three rates after
  the drift fix (was 0.72-0.51 before).

**Stated tolerance:** the invariance claim holds to within approximately 10% across the
supported sample-rate range (1000-4000 Hz) for EMG-representative input. The residual at high
sample rates and full sweat is a second-order interaction between the running-RMS estimator
and the drift term; it is small, does not affect the 1% symptom that motivated the fix, and is
declared an acceptable tolerance rather than a bug. If a tighter bound is required later, the
running-RMS estimator's sample-rate behavior is the place to investigate.

---

## Open validation items

Everything above is grounded in literature for its *form* and, where noted, its *magnitude*.
Several magnitudes are modeling choices not yet validated against real sweaty recordings. When
lab-recorded forearm EMG with sweat becomes available, the following should be calibrated
against it:

1. `max_attenuation` (Mechanism 1) against measured amplitude loss.
2. Sigmoid steepness k (Mechanism 1) against the measured amplitude-vs-sweat shape.
3. Whether a spectral shift occurs at all (Mechanism 2). If real data shows none, consistent
   with Abdoli-Eramaki, disable Mechanism 2 via `wet_cutoff_hz = dry_cutoff_hz`.
4. `drift_scale` (Mechanism 3) against measured baseline-drift amplitude.
5. Powerline harmonic weights (Mechanism 4) against a measured interference spectrum.

Citations still to verify against source PDFs before manuscript submission: De Luca et al.
2010, Webster 1984, Xu et al. 2020, Mewett et al. 2001.