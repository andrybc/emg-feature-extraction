"""
ninapro_to_calibration.py
=========================
Converts a NinaPro .mat file into a calibration CSV that the EMG Simulator
can import to replace its placeholder gesture patterns with real measured values.

The output CSV contains mean RMS amplitude per channel per gesture,
normalized so it plugs directly into GESTURE_CONFIG in the simulator.

Usage (command line):
    python ninapro_to_calibration.py path/to/S1_E1_A1.mat
    python ninapro_to_calibration.py path/to/S1_E1_A1.mat my_calibration.csv

Usage (imported by simulator):
    from ninapro_to_calibration import convert_mat_to_calibration_csv
    csv_path = convert_mat_to_calibration_csv('S1_E1_A1.mat')

Requirements:
    pip install numpy scipy pandas

NinaPro download:
    Register free at ninapro.hevs.ch
    Recommended starting point: DB1 or DB2, Exercise 1 files (S1_E1_A1.mat)
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.io as sio

# =============================================================================
# GESTURE LABEL MAPPING
# =============================================================================
# NinaPro uses integer labels to identify gestures.
# These integers differ between database versions and exercise files.
# Edit the values below to match whichever database file you are using.
#
# NinaPro DB1 -- Exercise 1 (S*_E1_A1.mat):
#   0  = Rest
#   17 = Open hand
#   18 = Closed hand (fist)
#   20 = Wrist flexion
#   21 = Wrist extension
#
# NinaPro DB2 -- Exercise 1 (S*_E1_A1.mat):
#   0  = Rest
#   13 = Open hand
#   14 = Closed hand (power grip)
#   15 = Wrist flexion
#   16 = Wrist extension
#
# NinaPro DB5 -- Exercise 1:
#   0  = Rest
#   7  = Open hand
#   8  = Closed hand
#   10 = Wrist flexion
#   11 = Wrist extension
# =============================================================================

# Default mapping: NinaPro DB1 Exercise 1
# Change these four integers to match your database
NINAPRO_GESTURE_MAP = {
    'Rest':            0,   # Always 0 in every NinaPro database
    'Hand Open':       17,  # DB1=17, DB2=13, DB5=7
    'Hand Closed':     18,  # DB1=18, DB2=14, DB5=8
    'Wrist Flexion':   20,  # DB1=20, DB2=15, DB5=10
    'Wrist Extension': 21,  # DB1=21, DB2=16, DB5=11
}

# Simulator gesture labels -- these never change
SIMULATOR_LABELS = {
    'Rest':            0,
    'Hand Open':       1,
    'Hand Closed':     2,
    'Wrist Flexion':   3,
    'Wrist Extension': 4,
}

# Number of EMG channels to extract
# NinaPro DB1 has 10 channels (8 EMG + 2 from force glove)
# We use only the first 8
N_CHANNELS = 8

# Assumed NinaPro sampling rate (Hz)
# DB1 and DB2: 2000 Hz
# DB5: 2000 Hz
NINAPRO_FS = 2000


# =============================================================================
# LOADING
# =============================================================================

def load_ninapro_mat(mat_path: str) -> dict:
    """
    Load a NinaPro .mat file and return EMG and label arrays.

    NinaPro .mat structure:
        emg          -- (n_samples, n_channels) float, raw EMG in mV
        restimulus   -- (n_samples, 1) int, gesture label per sample
                        ('re' prefix = re-labeled version with cleaner boundaries)
        rerepetition -- (n_samples, 1) int, repetition number
        stimulus     -- (n_samples, 1) int, original stimulus (noisier boundaries)
        repetition   -- (n_samples, 1) int, original repetition

    We use 'restimulus' rather than 'stimulus' because it has cleaner
    onset/offset boundaries, making RMS computation more accurate.

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

    # Confirm expected keys exist
    required_keys = ['emg', 'restimulus', 'rerepetition']
    missing = [k for k in required_keys if k not in mat]
    if missing:
        raise KeyError(
            f"Expected keys not found in .mat file: {missing}\n"
            f"Available keys: {[k for k in mat.keys() if not k.startswith('_')]}\n"
            f"Make sure this is a NinaPro file."
        )

    # Extract and flatten
    emg         = mat['emg'].astype(np.float64)            # (n_samples, n_ch)
    labels      = mat['restimulus'].flatten().astype(int)   # (n_samples,)
    repetitions = mat['rerepetition'].flatten().astype(int) # (n_samples,)

    duration_s = emg.shape[0] / NINAPRO_FS

    print(f"  EMG shape:      {emg.shape}  ({emg.shape[1]} channels)")
    print(f"  Total samples:  {emg.shape[0]}  ({duration_s:.1f}s at {NINAPRO_FS} Hz)")
    print(f"  Unique labels:  {sorted(np.unique(labels))}")
    print(f"  Repetitions:    {sorted(np.unique(repetitions))}")

    return {
        'emg':         emg,
        'labels':      labels,
        'repetitions': repetitions,
    }


# =============================================================================
# RMS COMPUTATION
# =============================================================================

def compute_calibration_table(
    data:        dict,
    gesture_map: dict,
    n_channels:  int = N_CHANNELS
) -> pd.DataFrame:
    """
    Compute mean RMS amplitude per channel per gesture.

    For each gesture:
        1. Find all samples where label == that gesture's NinaPro label
        2. Compute RMS per channel across those samples
        3. Normalize so the global peak channel == 1.0
        4. Store as one row in the output DataFrame

    Parameters
    ----------
    data        : dict -- output of load_ninapro_mat()
    gesture_map : dict -- {gesture_name: ninapro_label}
    n_channels  : int  -- how many channels to use (first n)

    Returns
    -------
    pd.DataFrame -- calibration table ready to save as CSV
    """
    # Use only the first n_channels (ignore force glove channels etc.)
    emg    = data['emg'][:, :n_channels]
    labels = data['labels']

    rows = []
    print(f"\nComputing mean RMS per channel per gesture ({n_channels} channels)...")

    for gesture_name, ninapro_label in gesture_map.items():
        sim_label = SIMULATOR_LABELS[gesture_name]

        # Boolean mask: True for every sample belonging to this gesture
        mask      = labels == ninapro_label
        n_samples = int(mask.sum())

        if n_samples == 0:
            print(
                f"  WARNING: No samples found for '{gesture_name}' "
                f"(NinaPro label {ninapro_label}).\n"
                f"  Check NINAPRO_GESTURE_MAP at the top of this script.\n"
                f"  Filling with zeros -- this gesture will not be calibrated."
            )
            ch_rms = np.zeros(n_channels)
        else:
            gesture_emg = emg[mask]   # (n_gesture_samples, n_channels)

            # RMS per channel:
            # For each column (channel), compute sqrt(mean(x^2)) across rows (time)
            # axis=0 collapses the time dimension, leaving (n_channels,)
            ch_rms = np.sqrt(np.mean(gesture_emg ** 2, axis=0))

            print(
                f"  {gesture_name:20s} | {n_samples:6,} samples | "
                f"peak_ch={ch_rms.argmax()+1}  "
                f"mean_rms={ch_rms.mean():.5f} mV  "
                f"max_rms={ch_rms.max():.5f} mV"
            )

        rows.append({
            'gesture_name':      gesture_name,
            'gesture_label':     sim_label,
            'ninapro_label':     ninapro_label,
            'n_samples':         n_samples,
            **{f'ch{i+1}_rms_raw': ch_rms[i] for i in range(n_channels)},
        })

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # NORMALIZATION
    # Divide all RMS values by the global maximum so the range is 0 to 1.
    # This is important because the simulator's pattern arrays are
    # weights between 0 and 1, not raw mV values.
    # The overall scale is then set by peak_amplitude separately.
    # ------------------------------------------------------------------
    raw_cols   = [f'ch{i+1}_rms_raw' for i in range(n_channels)]
    norm_cols  = [f'ch{i+1}_rms' for i in range(n_channels)]
    global_max = df[raw_cols].values.max()

    if global_max > 0:
        for i in range(n_channels):
            df[norm_cols[i]] = df[raw_cols[i]] / global_max
    else:
        for col in norm_cols:
            df[col] = 0.0

    # Peak amplitude per gesture = max normalized channel value
    # Rest is special: always set to a small fixed noise floor value
    df['peak_amplitude'] = df[norm_cols].max(axis=1)
    df.loc[df['gesture_name'] == 'Rest', 'peak_amplitude'] = 0.065

    # Drop the raw columns -- only keep normalized values in output
    df = df.drop(columns=raw_cols)

    return df


# =============================================================================
# SAVING
# =============================================================================

def save_calibration_csv(df: pd.DataFrame, output_path: str):
    """
    Save the calibration DataFrame to a CSV file.

    Columns saved:
        gesture_name    -- gesture string name
        gesture_label   -- integer label (0-4)
        ninapro_label   -- original NinaPro label used to find samples
        n_samples       -- how many EMG samples were averaged
        ch1_rms..ch8_rms -- normalized RMS per channel (0 to 1)
        peak_amplitude  -- overall gesture amplitude scale factor
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.8f')

    print(f"\nCalibration CSV saved: {output_path}")
    print(f"\nPreview (normalized values):")

    preview_cols = ['gesture_name'] + [f'ch{i+1}_rms' for i in range(N_CHANNELS)] + ['peak_amplitude']
    print(df[preview_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


# =============================================================================
# MAIN PIPELINE FUNCTION
# Called by both the command line entry point and the simulator app
# =============================================================================

def convert_mat_to_calibration_csv(
    mat_path:    str,
    output_path: str  = None,
    gesture_map: dict = None
) -> str:
    """
    Full pipeline: NinaPro .mat --> calibration CSV.

    Parameters
    ----------
    mat_path    : str  -- path to NinaPro .mat file
    output_path : str  -- where to save the CSV
                          default: same folder as .mat, with _calibration.csv suffix
    gesture_map : dict -- {gesture_name: ninapro_label}
                          default: NINAPRO_GESTURE_MAP (DB1 Exercise 1)

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

    Called by both this script and the simulator app when importing from CSV.

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

    # Validate expected columns
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

    print(f"Calibration CSV loaded: {csv_path}")
    print(f"  Gestures found: {list(df['gesture_name'])}")

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