"""
ninapro_to_calibration.py
=========================
Converts a NinaPro .mat file into a calibration CSV that the EMG Simulator
can import to replace its placeholder gesture patterns with real measured values.

The output CSV contains:
    - Mean RMS amplitude per channel per gesture (normalized to [0, 1])
    - Per-channel standard deviation across repetitions (also normalized)
    - Peak amplitude per gesture (the overall loudness scalar)
    - Original mV scale (so the simulator can report absolute units)

The CSV plugs directly into GESTURE_CONFIG.pattern, GESTURE_CONFIG.pattern_std,
and GESTURE_CONFIG.amplitude in the simulator.

Usage (command line):
    python ninapro_to_calibration.py path/to/S1_E1_A1.mat
    python ninapro_to_calibration.py path/to/S1_E1_A1.mat my_calibration.csv

Usage (imported by simulator):
    from ninapro_to_calibration import convert_mat_to_calibration_csv
    csv_path = convert_mat_to_calibration_csv('S1_E1_A1.mat')

Requirements:
    pip install numpy scipy pandas

NinaPro download:
    Register free at https://ninapro.hevs.ch/
    Verified-working starting point: DB2 subject 1 file S1_E1_A1.mat
    (Exercise B: 17 hand and wrist movements, 12-channel raw EMG at 2 kHz)
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.io as sio

# =============================================================================
# GESTURE LABEL MAPPING
# =============================================================================
# IMPORTANT: NinaPro databases use PER-FILE label integers, NOT the global
# 1-52 numbering that appears in many published papers (where Exercise A is
# movements 1-12, B is 13-29, C is 30-52). Each individual .mat file stores
# labels starting from 0 (rest) and counting up within that file's exercise.
#
# This is the source of a lot of confusion. The values below are VERIFIED
# against S1_E1_A1.mat from DB2 by inspecting per-channel RMS patterns:
# wrist flexion (label 13) shows palm-side activation (CH3, CH4); wrist
# extension (label 14) shows dorsal-side activation (CH7, CH8). Anatomy
# matches expectation, so the labels are correct.
#
# For DB2 file _E1_A1.mat (Exercise B: 17 hand + wrist movements):
#
#   Movement order follows Atzori et al. 2014, Sci Data, Fig. 2:
#     0: Rest
#     1: Thumb up
#     2: Extension of index and middle, flexion of others
#     3: Flexion of ring and little, extension of others
#     4: Thumb opposing base of little finger
#     5: Abduction of all fingers          <-- closest DB2 "Hand Open"
#     6: Fingers flexed together in fist   <-- DB2 "Hand Closed"
#     7: Pointing index
#     8: Adduction of extended fingers
#     9-12: Wrist supination/pronation (middle/little finger axis)
#    13: Wrist flexion                     <-- DB2 "Wrist Flexion"
#    14: Wrist extension                   <-- DB2 "Wrist Extension"
#    15: Wrist radial deviation
#    16: Wrist ulnar deviation
#    17: Wrist extension with closed hand
#
# Note: Exercise B does NOT contain a standalone "open hand" gesture
# (relaxed hand extension). The closest analogue is label 5 ("Abduction
# of all fingers"), which captures active extensor recruitment, the
# physiological signature usually meant by "hand open" in myoelectric
# literature. Label 8 ("Adduction of extended fingers") is a defensible
# alternative if you want a flat-hand semantics instead.
#
# For other databases the file labels almost certainly differ. Do not
# trust the integers below for DB1, DB3, DB4, DB5, DB6, or DB7 without
# first running an inspection script on a sample .mat file to verify.
# =============================================================================

# DB2 Exercise B (S*_E1_A1.mat) -- VERIFIED via per-channel RMS inspection.
NINAPRO_GESTURE_MAP = {
    'Rest':            0,
    'Hand Open':       5,    # "Abduction of all fingers"
    'Hand Closed':     6,    # "Fingers flexed together in fist"
    'Wrist Flexion':   13,
    'Wrist Extension': 14,
}

# Simulator gesture labels -- these never change
SIMULATOR_LABELS = {
    'Rest':            0,
    'Hand Open':       1,
    'Hand Closed':     2,
    'Wrist Flexion':   3,
    'Wrist Extension': 4,
}

# Number of EMG channels to extract.
# DB2 records 12 channels with the Delsys Trigno system. The first 8 are
# deliberately the equally-spaced ring around the forearm at the radio-
# humeral joint (the armband-equivalent subset, per Atzori 2014). The
# remaining 4 are targeted electrodes over flexor/extensor digitorum,
# biceps, and triceps. Taking the first 8 gives the armband-equivalent
# pattern we want for an 8-channel simulator.
N_CHANNELS = 8

# Assumed NinaPro sampling rate (Hz).
# DB2 / DB3 / DB4 / DB6 / DB7 / DB10: 2000 Hz (raw Trigno / Cometa output)
# DB1:  100 Hz (Otto Bock rectified envelope -- NOT raw EMG, avoid for RMS calibration)
# DB5:  200 Hz (Myo armband)
NINAPRO_FS = 2000

# Rest's per-channel "variability" is qualitatively different from a held
# gesture's: there is no real trial-to-trial muscle variability when no
# muscle is contracting. We assign Rest a small fixed CV so the simulator's
# per-click variability sampler does something sensible but quiet.
REST_PATTERN_CV = 0.05  # 5% of mean per channel


# =============================================================================
# LOADING
# =============================================================================

def load_ninapro_mat(mat_path: str) -> dict:
    """
    Load a NinaPro .mat file and return EMG, label, and repetition arrays.

    NinaPro .mat structure (relevant keys):
        emg          -- (n_samples, n_channels) float, raw EMG in Volts
        restimulus   -- (n_samples, 1) int, gesture label per sample
                        ('re' prefix = re-labeled with cleaner onset/offset)
        rerepetition -- (n_samples, 1) int, repetition number (1-6 typically,
                        0 during rest periods)
        stimulus     -- (n_samples, 1) int, original noisier labels
        repetition   -- (n_samples, 1) int, original repetition

    We use 'restimulus' rather than 'stimulus' because the re-labeled
    version uses movement detection to refine boundaries, making per-rep
    RMS computation more accurate.

    Parameters
    ----------
    mat_path : str -- full path to the .mat file

    Returns
    -------
    dict with keys: emg, labels, repetitions
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")

    print(f"Loading: {mat_path}")

    try:
        mat = sio.loadmat(mat_path)
    except Exception as e:
        raise RuntimeError(
            f"Could not load .mat file.\n"
            f"Make sure scipy is installed: pip install scipy\n"
            f"Error: {e}"
        )

    # Confirm expected keys exist before trying to use them.
    required_keys = ['emg', 'restimulus', 'rerepetition']
    missing = [k for k in required_keys if k not in mat]
    if missing:
        raise KeyError(
            f"Expected keys not found in .mat file: {missing}\n"
            f"Available keys: {[k for k in mat.keys() if not k.startswith('_')]}\n"
            f"Make sure this is a NinaPro file."
        )

    # Extract and flatten. NinaPro stores label/repetition as column vectors;
    # flatten makes them 1D for easier masking.
    emg         = mat['emg'].astype(np.float64)             # (n_samples, n_ch)
    labels      = mat['restimulus'].flatten().astype(int)   # (n_samples,)
    repetitions = mat['rerepetition'].flatten().astype(int) # (n_samples,)

    # restimulus is sometimes one sample shorter than emg (last sample dropped
    # during relabeling). Trim emg to match so downstream masking is clean.
    n_align = min(emg.shape[0], len(labels), len(repetitions))
    emg         = emg[:n_align]
    labels      = labels[:n_align]
    repetitions = repetitions[:n_align]

    duration_s = emg.shape[0] / NINAPRO_FS

    print(f"  EMG shape:          {emg.shape}  ({emg.shape[1]} channels)")
    print(f"  Total samples:      {emg.shape[0]:,}  ({duration_s:.1f}s at {NINAPRO_FS} Hz)")
    print(f"  Unique labels:      {sorted(np.unique(labels))}")
    print(f"  Repetitions found:  {sorted(np.unique(repetitions))}")

    return {
        'emg':         emg,
        'labels':      labels,
        'repetitions': repetitions,
    }


# =============================================================================
# PER-REPETITION RMS COMPUTATION
# =============================================================================
# This is the substantive computational change vs. the previous script.
#
# OLD approach: pool ALL samples of a gesture together, compute one RMS
# per channel. Result: a single mean pattern, no variability info.
#
# NEW approach: group samples by (gesture, repetition), compute RMS per
# channel per repetition. With ~6 repetitions per gesture in DB2, we
# get a 6-by-8 matrix per gesture. Mean across reps gives the pattern.
# Std across reps gives the per-channel trial-to-trial variability, which
# is exactly what the simulator's pattern_std field models.
#
# Why this matters: Kapelner et al. 2018 and Phinyomark et al. 2012 both
# report 10-20% coefficient of variation in per-channel RMS across
# repetitions of the same gesture, even within a single session by a
# single subject. Without measuring this, the simulator's variability
# model is just a guess.
# =============================================================================

def compute_per_rep_rms(
    emg:         np.ndarray,
    labels:      np.ndarray,
    repetitions: np.ndarray,
    target_lbl:  int,
    n_channels:  int,
) -> np.ndarray:
    """
    Compute per-channel RMS for each repetition of a single gesture.

    Parameters
    ----------
    emg         : (N, n_ch) raw EMG
    labels      : (N,) restimulus values
    repetitions : (N,) rerepetition values
    target_lbl  : the gesture's NinaPro label
    n_channels  : how many channels to compute over

    Returns
    -------
    rms_per_rep : (n_reps, n_channels) array, one row per rep, OR
                  zero-rows array if no samples match.
                  n_reps is the number of distinct non-zero reps found
                  containing this gesture (typically 6 for DB2).
    """
    gesture_mask = labels == target_lbl

    if not gesture_mask.any():
        return np.zeros((0, n_channels))

    # Repetition numbers that appear together with this gesture's label.
    # We drop rep 0 because that's the rest-between-movements rep and
    # doesn't represent a contraction trial.
    reps_for_gesture = sorted(set(repetitions[gesture_mask]) - {0})

    if not reps_for_gesture:
        # The label exists but all samples are tagged rep 0 (this happens
        # for Rest itself, where rerepetition is 0 by convention).
        # Fall through to whole-segment RMS so Rest still gets a pattern.
        seg = emg[gesture_mask, :n_channels]
        return np.sqrt(np.mean(seg**2, axis=0))[None, :]   # shape (1, n_ch)

    per_rep_rms = []
    for rep in reps_for_gesture:
        rep_mask = gesture_mask & (repetitions == rep)
        seg      = emg[rep_mask, :n_channels]
        if seg.shape[0] < 100:   # need at least ~50 ms of data at 2 kHz
            continue
        ch_rms = np.sqrt(np.mean(seg**2, axis=0))
        per_rep_rms.append(ch_rms)

    return np.array(per_rep_rms) if per_rep_rms else np.zeros((0, n_channels))


def compute_calibration_table(
    data:        dict,
    gesture_map: dict,
    n_channels:  int = N_CHANNELS,
) -> pd.DataFrame:
    """
    Compute mean RMS and STD-across-repetitions per channel per gesture.

    For each gesture in gesture_map:
      1. Find each repetition (typically 6) of that gesture.
      2. Compute per-channel RMS within each repetition.
      3. Take mean and std across repetitions.
      4. Normalize across all gestures by the global mean-RMS maximum.
      5. Track the original mV scale so the simulator can report it.

    Parameters
    ----------
    data        : dict -- output of load_ninapro_mat()
    gesture_map : dict -- {gesture_name: ninapro_label}
    n_channels  : int  -- how many channels to use (first n)

    Returns
    -------
    pd.DataFrame -- calibration table ready to save as CSV
    """
    emg         = data['emg']
    labels      = data['labels']
    repetitions = data['repetitions']

    rows = []
    print(f"\nComputing per-repetition RMS per channel per gesture "
          f"({n_channels} channels)...")
    print(f"  {'gesture':<20} {'reps':>5}  {'mean_rms':>10}  {'std_rms':>10}  peak_ch")
    print("  " + "-" * 68)

    for gesture_name, ninapro_label in gesture_map.items():
        sim_label = SIMULATOR_LABELS[gesture_name]

        rms_per_rep = compute_per_rep_rms(
            emg, labels, repetitions, ninapro_label, n_channels
        )
        n_reps = rms_per_rep.shape[0]

        if n_reps == 0:
            print(
                f"  WARNING: No samples found for '{gesture_name}' "
                f"(NinaPro label {ninapro_label}). Filling with zeros.\n"
                f"  Check NINAPRO_GESTURE_MAP at the top of this script."
            )
            ch_mean = np.zeros(n_channels)
            ch_std  = np.zeros(n_channels)
        else:
            # Average across reps gives the canonical per-channel pattern.
            ch_mean = rms_per_rep.mean(axis=0)
            # Std across reps gives the trial-to-trial variability.
            # ddof=0: population std (we're treating these reps as the
            # full population, not a sample). ddof=1 would inflate std
            # for small n by Bessel's correction, which we don't want.
            # If there's only 1 rep (Rest fallback), std is 0.
            if n_reps >= 2:
                ch_std = rms_per_rep.std(axis=0, ddof=0)
            else:
                # Single rep: assign a small fraction of the mean as a
                # placeholder variability so the simulator's per-click
                # sampling still does something sensible.
                ch_std = REST_PATTERN_CV * ch_mean

            print(
                f"  {gesture_name:<20} {n_reps:>5}  {ch_mean.mean():>10.6f}  "
                f"{ch_std.mean():>10.6f}  CH{ch_mean.argmax()+1}"
            )

        row = {
            'gesture_name':  gesture_name,
            'gesture_label': sim_label,
            'ninapro_label': ninapro_label,
            'n_reps':        n_reps,
        }
        # Raw (un-normalized) values in original mV/V units. We compute
        # the normalization factor after collecting all gestures, so we
        # keep the raw values until we know the global max.
        for i in range(n_channels):
            row[f'ch{i+1}_rms_raw'] = ch_mean[i]
            row[f'ch{i+1}_std_raw'] = ch_std[i]
        rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # NORMALIZATION TO [0, 1]
    #
    # The simulator's pattern arrays are dimensionless weights in [0, 1],
    # with overall loudness set by the separate peak_amplitude scalar.
    # We divide both the mean and the std by the SAME global maximum,
    # so they stay in the same units relative to each other.
    #
    # Dividing the std by the mean's max (not by each gesture's own max)
    # preserves the inter-gesture amplitude ratios. A gesture that's
    # quieter overall will have a proportionally smaller std after
    # normalization, which is the right behavior.
    # ------------------------------------------------------------------
    raw_mean_cols  = [f'ch{i+1}_rms_raw' for i in range(n_channels)]
    raw_std_cols   = [f'ch{i+1}_std_raw' for i in range(n_channels)]
    norm_mean_cols = [f'ch{i+1}_rms'     for i in range(n_channels)]
    norm_std_cols  = [f'ch{i+1}_std'     for i in range(n_channels)]

    global_max = df[raw_mean_cols].values.max()
    original_peak_mV = float(global_max) * 1000.0  # NinaPro EMG is stored in Volts

    if global_max > 0:
        for i in range(n_channels):
            df[norm_mean_cols[i]] = df[raw_mean_cols[i]] / global_max
            df[norm_std_cols[i]]  = df[raw_std_cols[i]]  / global_max
    else:
        # Defensive zero-out if the entire file was silent.
        for col in norm_mean_cols + norm_std_cols:
            df[col] = 0.0

    # Peak amplitude per gesture: max normalized channel value.
    # The simulator's auto-normalize branch reads this column.
    df['peak_amplitude'] = df[norm_mean_cols].max(axis=1)

    # Rest is special. Its measured RMS is a noise floor, not a contraction
    # level. Setting a fixed small peak_amplitude here keeps the simulator's
    # rest "noise floor" consistent regardless of which subject was calibrated.
    df.loc[df['gesture_name'] == 'Rest', 'peak_amplitude'] = 0.065

    # ------------------------------------------------------------------
    # SCALE METADATA
    #
    # The normalization above destroys absolute-units information. We
    # preserve it as a column on every row (same value, since it's a
    # file-level metadata). The simulator's _apply_calibration method
    # can read this column and populate self.calibration_scale_mV so
    # downstream users still know what the [0, 1] range maps to in mV.
    #
    # We write peak in mV (not Volts) because that's the unit human
    # readers and most EMG literature use.
    # ------------------------------------------------------------------
    df['original_peak_mV'] = original_peak_mV

    # Drop the raw columns -- only keep normalized values and metadata.
    df = df.drop(columns=raw_mean_cols + raw_std_cols)

    print(f"\n  Original peak mV: {original_peak_mV:.4f} mV")
    print(f"  All channels normalized to [0, 1] using this peak.")

    return df


# =============================================================================
# SAVING
# =============================================================================

def save_calibration_csv(df: pd.DataFrame, output_path: str):
    """
    Save the calibration DataFrame to a CSV file.

    Columns saved:
        gesture_name      -- gesture string name
        gesture_label     -- integer label (0-4)
        ninapro_label     -- original NinaPro label used to find samples
        n_reps            -- number of repetitions averaged
        ch1_rms..ch8_rms  -- normalized per-channel mean RMS in [0, 1]
        ch1_std..ch8_std  -- normalized per-channel std across reps in [0, 1]
        peak_amplitude    -- overall gesture amplitude scale factor
        original_peak_mV  -- original mV scale before normalization
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.8f')

    print(f"\nCalibration CSV saved: {output_path}")
    print(f"\nPreview (normalized means):")

    preview_cols = (
        ['gesture_name', 'n_reps']
        + [f'ch{i+1}_rms' for i in range(N_CHANNELS)]
        + ['peak_amplitude']
    )
    print(df[preview_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print(f"\nPreview (normalized stds across reps):")
    std_cols = ['gesture_name'] + [f'ch{i+1}_std' for i in range(N_CHANNELS)]
    print(df[std_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


# =============================================================================
# MAIN PIPELINE FUNCTION
# Called by both the command-line entry point and the simulator app
# =============================================================================

def convert_mat_to_calibration_csv(
    mat_path:    str,
    output_path: str  = None,
    gesture_map: dict = None,
) -> str:
    """
    Full pipeline: NinaPro .mat --> calibration CSV.

    Parameters
    ----------
    mat_path    : str  -- path to NinaPro .mat file
    output_path : str  -- where to save the CSV
                          default: same folder as .mat, suffix _calibration.csv
    gesture_map : dict -- {gesture_name: ninapro_label}
                          default: NINAPRO_GESTURE_MAP (DB2 Exercise B)

    Returns
    -------
    str -- path to the saved calibration CSV
    """
    if gesture_map is None:
        gesture_map = NINAPRO_GESTURE_MAP

    if output_path is None:
        base        = os.path.splitext(mat_path)[0]
        output_path = base + '_calibration.csv'

    data = load_ninapro_mat(mat_path)
    df   = compute_calibration_table(data, gesture_map)
    save_calibration_csv(df, output_path)

    return output_path


def load_calibration_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a calibration CSV and return a validated DataFrame.

    Called by this script and by the simulator app when importing from CSV.

    Parameters
    ----------
    csv_path : str -- path to a calibration CSV

    Returns
    -------
    pd.DataFrame -- validated calibration table
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Required columns: name, label, peak amplitude, and the 8 per-channel
    # mean RMS columns. Std columns and original_peak_mV are optional and
    # only present if the CSV was generated by this updated script.
    required = ['gesture_name', 'gesture_label', 'peak_amplitude'] + \
               [f'ch{i+1}_rms' for i in range(N_CHANNELS)]
    missing  = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Calibration CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Make sure this file was generated by the EMG Simulator "
            f"or ninapro_to_calibration.py"
        )

    has_std   = all(f'ch{i+1}_std' in df.columns for i in range(N_CHANNELS))
    has_scale = 'original_peak_mV' in df.columns

    print(f"Calibration CSV loaded: {csv_path}")
    print(f"  Gestures found: {list(df['gesture_name'])}")
    print(f"  Has per-channel std columns: {has_std}")
    print(f"  Has absolute mV scale:       {has_scale}")
    if has_scale and len(df) > 0:
        print(f"  Original peak mV:            {df['original_peak_mV'].iloc[0]:.4f} mV")

    return df


# =============================================================================
# COMMAND LINE ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        print("No file provided.")
        print("Usage: python ninapro_to_calibration.py path/to/S1_E1_A1.mat [output.csv]")
        sys.exit(1)

    mat_path    = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result_path = convert_mat_to_calibration_csv(mat_path, output_path)

    print(f"\nDone.")
    print(f"Import into the EMG Simulator:")
    print(f"  Click 'Import Calibration' --> 'From CSV' --> select: {result_path}")