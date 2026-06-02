"""
apply_sweat_offline.py
======================
Apply the SweatProcessor to a recorded EMG CSV file. The same class that
runs live in the simulator can be used as a batch tool to generate
synthetic-artifact datasets from clean recordings.

Three modes:

  constant     Apply one sweat level to the entire recording.
  sweep        Linearly sweep sweat from 0 to 1 across the recording.
  multi-seed   Apply the same level multiple times with different seeds.

Usage examples (run from the simulator's directory):

  python apply_sweat_offline.py constant  input.csv output.csv --level 0.7
  python apply_sweat_offline.py sweep     input.csv output.csv
  python apply_sweat_offline.py multi-seed input.csv outputs/ --level 0.5 --n-seeds 5
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts.sweat import SweatProcessor


N_CHANNELS = 8


# ----------------------------------------------------------------------
# Loading and saving
# ----------------------------------------------------------------------

def load_recording(path: str):
    """
    Load a CSV produced by the simulator. Returns:

        df    : the full DataFrame (so we can write back with metadata)
        emg   : np.ndarray, shape (n_samples, 8), the channel data
        fs    : float, detected sample rate in Hz
    """
    df = pd.read_csv(path)

    # Pull channel columns; the simulator writes them as 'ch1_mV' ... 'ch8_mV'
    ch_cols = [f'ch{i+1}_mV' for i in range(N_CHANNELS)]
    missing = [c for c in ch_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected channel columns: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )

    emg = df[ch_cols].values.astype(np.float64)

    # Detect sample rate from the timestamp column
    if 'timestamp_s' in df.columns:
        dt = np.diff(df['timestamp_s'].values)
        fs = 1.0 / np.median(dt)
    else:
        # Default if timestamps are missing
        fs = 1000.0
        print(f"  warning: no timestamp_s column, assuming fs={fs} Hz")

    return df, emg, fs


def save_recording(df_template: pd.DataFrame, emg_new: np.ndarray, path: str):
    """
    Write a modified EMG array back to disk, preserving all non-channel
    columns from the template DataFrame.
    """
    df_out = df_template.copy()
    for i in range(N_CHANNELS):
        df_out[f'ch{i+1}_mV'] = emg_new[:, i]
    df_out.to_csv(path, index=False)


# ----------------------------------------------------------------------
# Three processing modes
# ----------------------------------------------------------------------

def apply_constant(
    input_csv: str,
    output_csv: str,
    sweat_level: float,
    seed: int = 42,
) -> None:
    """Apply one sweat level to the entire recording."""
    print(f"Loading {input_csv}")
    df, emg, fs = load_recording(input_csv)
    print(f"  {len(emg)} samples, {fs:.1f} Hz, duration {len(emg)/fs:.2f}s")

    print(f"Applying constant sweat level: {sweat_level:.2f}")
    sp = SweatProcessor(
        n_channels=N_CHANNELS,
        fs=fs,
        tau_sweat=0.0,        # offline: snap to target
        seed=seed,
    )
    sp.set_sweat_level(sweat_level)
    emg_out = sp.process(emg)

    print(f"Saving {output_csv}")
    save_recording(df, emg_out, output_csv)
    print("Done.")


def apply_sweep(
    input_csv: str,
    output_csv: str,
    start_level: float = 0.0,
    end_level: float = 1.0,
    seed: int = 42,
    chunk_size: int = 1000,
) -> None:
    """
    Sweep sweat level linearly from start to end across the recording.

    We can't change sweat mid-buffer (the processor uses one level for
    each process() call), so we feed the recording in chunks and update
    the level between chunks. With chunk_size=1000 at fs=1000, the level
    updates once per second of signal.
    """
    print(f"Loading {input_csv}")
    df, emg, fs = load_recording(input_csv)
    n_samples = len(emg)
    print(f"  {n_samples} samples, {fs:.1f} Hz")

    print(f"Sweeping sweat: {start_level:.2f} -> {end_level:.2f}")
    sp = SweatProcessor(
        n_channels=N_CHANNELS,
        fs=fs,
        tau_sweat=0.0,
        seed=seed,
    )

    emg_out = np.empty_like(emg)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        start = ci * chunk_size
        end   = min(start + chunk_size, n_samples)

        # Linear interpolation: where in the sweep is this chunk?
        frac  = ci / max(n_chunks - 1, 1)
        level = start_level + (end_level - start_level) * frac
        sp.set_sweat_level(level)

        emg_out[start:end] = sp.process(emg[start:end])

    print(f"Saving {output_csv}")
    save_recording(df, emg_out, output_csv)
    print("Done.")


def apply_multi_seed(
    input_csv: str,
    output_dir: str,
    sweat_level: float,
    n_seeds: int = 5,
    base_seed: int = 0,
) -> None:
    """
    Apply the same sweat level multiple times with different seeds,
    producing N versions of the contaminated recording. Useful for
    ensemble experiments where you want the same gesture data with
    independent artifact realizations.
    """
    print(f"Loading {input_csv}")
    df, emg, fs = load_recording(input_csv)
    print(f"  {len(emg)} samples, {fs:.1f} Hz")

    os.makedirs(output_dir, exist_ok=True)
    base = Path(input_csv).stem  # filename without extension

    for i in range(n_seeds):
        seed = base_seed + i
        sp = SweatProcessor(
            n_channels=N_CHANNELS,
            fs=fs,
            tau_sweat=0.0,
            seed=seed,
        )
        sp.set_sweat_level(sweat_level)
        emg_out = sp.process(emg)

        out_path = os.path.join(
            output_dir,
            f"{base}_sweat{int(sweat_level*100):03d}_seed{seed}.csv"
        )
        save_recording(df, emg_out, out_path)
        print(f"  Saved seed {seed}: {out_path}")

    print(f"Done. {n_seeds} files in {output_dir}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply SweatProcessor to a recorded EMG CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # constant
    p_const = subparsers.add_parser('constant', help='Apply one sweat level')
    p_const.add_argument('input',  help='Input CSV path')
    p_const.add_argument('output', help='Output CSV path')
    p_const.add_argument('--level', type=float, required=True,
                         help='Sweat level in [0, 1]')
    p_const.add_argument('--seed', type=int, default=42)

    # sweep
    p_sweep = subparsers.add_parser('sweep', help='Linearly sweep sweat level')
    p_sweep.add_argument('input',  help='Input CSV path')
    p_sweep.add_argument('output', help='Output CSV path')
    p_sweep.add_argument('--start', type=float, default=0.0)
    p_sweep.add_argument('--end',   type=float, default=1.0)
    p_sweep.add_argument('--seed',  type=int, default=42)
    p_sweep.add_argument('--chunk-size', type=int, default=1000,
                         help='Samples per chunk (controls sweep granularity)')

    # multi-seed
    p_multi = subparsers.add_parser('multi-seed',
                                    help='Apply same level with multiple seeds')
    p_multi.add_argument('input',  help='Input CSV path')
    p_multi.add_argument('output', help='Output directory path')
    p_multi.add_argument('--level', type=float, required=True)
    p_multi.add_argument('--n-seeds',   type=int, default=5)
    p_multi.add_argument('--base-seed', type=int, default=0)

    args = parser.parse_args()

    if args.mode == 'constant':
        apply_constant(args.input, args.output, args.level, args.seed)
    elif args.mode == 'sweep':
        apply_sweep(args.input, args.output,
                    args.start, args.end, args.seed, args.chunk_size)
    elif args.mode == 'multi-seed':
        apply_multi_seed(args.input, args.output,
                         args.level, args.n_seeds, args.base_seed)


if __name__ == '__main__':
    main()