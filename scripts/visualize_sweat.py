"""
visualize_sweat.py
==================
Visualize the effect of SweatProcessor on EMG recordings. Three modes:

  compare    Side-by-side plot of a clean vs contaminated recording.
  sweep      Spectrogram of a sweep output with theoretical cutoff overlay.
  ensemble   Overlay multiple multi-seed outputs on one channel.

Usage examples (from the simulator's directory):

  python visualize_sweat.py compare \\
      data/simulated/emg_rest_20260602_112645.csv \\
      data/simulated/emg_rest_20260602_112645_sweat70.csv

  python visualize_sweat.py sweep \\
      data/simulated/emg_rest_20260602_112645_sweep.csv

  python visualize_sweat.py ensemble \\
      data/simulated/ensemble/ --channel 5
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as dsp


N_CHANNELS = 8

# Color per channel, matching the simulator's palette
CHANNEL_COLORS = [
    '#34d399', '#38bdf8', '#fb923c', '#f472b6',
    '#a78bfa', '#fbbf24', '#4ade80', '#f87171',
]


# ----------------------------------------------------------------------
# Loading helper
# ----------------------------------------------------------------------

def load_recording(path):
    """Return (df, emg_array, fs) for any simulator CSV."""
    df = pd.read_csv(path)
    ch_cols = [f'ch{i+1}_mV' for i in range(N_CHANNELS)]
    emg = df[ch_cols].values.astype(np.float64)
    t   = df['timestamp_s'].values - df['timestamp_s'].iloc[0]
    fs  = 1.0 / np.median(np.diff(df['timestamp_s'].values))
    return df, emg, t, fs


# ----------------------------------------------------------------------
# Mode 1: compare two recordings
# ----------------------------------------------------------------------

def mode_compare(clean_path, contaminated_path, save_path=None):
    """
    Side-by-side plot: clean (left) vs contaminated (right) on all 8 channels,
    plus a spectrum comparison on the channel with the biggest amplitude.
    """
    print(f"Loading clean        : {clean_path}")
    df_c, emg_c, t, fs = load_recording(clean_path)
    print(f"Loading contaminated : {contaminated_path}")
    df_x, emg_x, _, _  = load_recording(contaminated_path)

    if emg_c.shape != emg_x.shape:
        raise ValueError(
            f"Shape mismatch: clean is {emg_c.shape}, "
            f"contaminated is {emg_x.shape}. They must be the same recording."
        )

    # Figure layout: 8 channel pairs (time-domain) on the left, plus a
    # spectrum comparison panel on the right
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(8, 3, width_ratios=[2, 2, 1.5], wspace=0.25, hspace=0.4)

    # Per-channel time domain
    for ch in range(N_CHANNELS):
        ax_clean = fig.add_subplot(gs[ch, 0])
        ax_dirty = fig.add_subplot(gs[ch, 1], sharey=ax_clean)

        color = CHANNEL_COLORS[ch]
        ax_clean.plot(t, emg_c[:, ch], linewidth=0.4, color=color)
        ax_dirty.plot(t, emg_x[:, ch], linewidth=0.4, color=color, alpha=0.85)

        ax_clean.set_ylabel(f'CH{ch+1}', rotation=0, labelpad=18, fontsize=9)
        for ax in (ax_clean, ax_dirty):
            ax.axhline(0, color='gray', linewidth=0.3, linestyle='--')
            ax.set_ylim(-2, 2)
            ax.grid(alpha=0.15)
            if ch < N_CHANNELS - 1:
                ax.set_xticklabels([])

        if ch == 0:
            ax_clean.set_title('CLEAN (input)', fontsize=11, color='#34d399')
            ax_dirty.set_title('CONTAMINATED (output)', fontsize=11, color='#f87171')

    fig.axes[-2].set_xlabel('Time (s)')
    fig.axes[-1].set_xlabel('Time (s)')

    # Find the channel with biggest amplitude for spectrum comparison
    ch_amp = [np.sqrt(np.mean(emg_c[:, c]**2)) for c in range(N_CHANNELS)]
    best_ch = int(np.argmax(ch_amp))

    # Spectrum panel
    ax_spec = fig.add_subplot(gs[0:4, 2])
    freqs   = np.fft.rfftfreq(len(t), 1/fs)
    spec_c  = np.abs(np.fft.rfft(emg_c[:, best_ch]))
    spec_x  = np.abs(np.fft.rfft(emg_x[:, best_ch]))
    ax_spec.semilogy(freqs, spec_c, linewidth=0.6, color='#34d399', label='clean')
    ax_spec.semilogy(freqs, spec_x, linewidth=0.6, color='#f87171', label='contaminated')
    ax_spec.axvline(60, color='red', linewidth=0.4, alpha=0.5, linestyle=':')
    ax_spec.text(62, ax_spec.get_ylim()[1] * 0.5, '60 Hz', fontsize=8, color='red')
    ax_spec.set_xlim(0, fs / 2)
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('|FFT|')
    ax_spec.set_title(f'Spectrum (CH{best_ch+1})', fontsize=10)
    ax_spec.grid(alpha=0.2)
    ax_spec.legend(fontsize=8, loc='upper right')

    # RMS ratio panel
    ax_rms = fig.add_subplot(gs[4:8, 2])
    rms_c  = np.array([np.sqrt(np.mean(emg_c[:, c]**2)) for c in range(N_CHANNELS)])
    rms_x  = np.array([np.sqrt(np.mean(emg_x[:, c]**2)) for c in range(N_CHANNELS)])
    channels = np.arange(1, N_CHANNELS + 1)
    width = 0.4
    ax_rms.bar(channels - width/2, rms_c, width, label='clean',        color='#34d399')
    ax_rms.bar(channels + width/2, rms_x, width, label='contaminated', color='#f87171', alpha=0.85)
    ax_rms.set_xlabel('Channel')
    ax_rms.set_ylabel('RMS (mV)')
    ax_rms.set_title('Per-channel RMS', fontsize=10)
    ax_rms.set_xticks(channels)
    ax_rms.legend(fontsize=8)
    ax_rms.grid(alpha=0.2, axis='y')

    plt.suptitle(
        f"Sweat artifact comparison: {os.path.basename(clean_path)}",
        y=0.995, fontsize=11,
    )

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# Mode 2: visualize a sweep recording
# ----------------------------------------------------------------------

def mode_sweep(sweep_path, channel=5, start_level=0.0, end_level=1.0, save_path=None):
    """
    Spectrogram of a sweep recording, with the theoretical lowpass cutoff
    overlaid. The cutoff curve should match where high-frequency energy
    drops off in the spectrogram.
    """
    print(f"Loading sweep: {sweep_path}")
    df, emg, t, fs = load_recording(sweep_path)
    ch_idx = channel - 1
    sig = emg[:, ch_idx]
    print(f"  {len(sig)} samples at {fs:.0f} Hz, plotting CH{channel}")

    # The sweat module's cutoff formula
    dry_cutoff = 400.0
    wet_cutoff = 150.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Time domain on top
    axes[0].plot(t, sig, linewidth=0.4, color=CHANNEL_COLORS[ch_idx])
    axes[0].set_ylabel(f'CH{channel} mV')
    axes[0].set_title(f'Sweep recording: CH{channel} time domain')
    axes[0].axhline(0, color='gray', linewidth=0.3, linestyle='--')
    axes[0].grid(alpha=0.2)
    axes[0].set_ylim(-2, 2)

    # Spectrogram on bottom
    f, ts, Sxx = dsp.spectrogram(sig, fs=fs, nperseg=256, noverlap=128)
    axes[1].pcolormesh(
        ts, f, 10 * np.log10(Sxx + 1e-12),
        shading='auto', cmap='viridis', vmin=-80, vmax=-20,
    )
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim(0, fs / 2)

    # Theoretical cutoff overlay
    duration = t[-1]
    sweat_at_ts = start_level + (end_level - start_level) * (ts / duration)
    sweat_at_ts = np.clip(sweat_at_ts, 0.0, 1.0)
    cutoff_theory = dry_cutoff + (wet_cutoff - dry_cutoff) * sweat_at_ts
    axes[1].plot(ts, cutoff_theory, color='red', linewidth=2,
                 linestyle='--', label='theoretical lowpass cutoff')
    axes[1].legend(loc='upper right', fontsize=9)

    axes[1].set_xlabel('Time (s)')
    axes[1].set_title(f'Spectrogram with theoretical cutoff overlay')

    plt.suptitle(f"Sweep visualization: {os.path.basename(sweep_path)}", y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# Mode 3: overlay multiple multi-seed outputs
# ----------------------------------------------------------------------

def mode_ensemble(ensemble_dir, channel=5, save_path=None):
    """
    Overlay the same channel from multiple multi-seed outputs, so you
    can see how the random-walk drift differs across seeds.
    """
    pattern = os.path.join(ensemble_dir, '*seed*.csv')
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}'. Check the directory path."
        )
    print(f"Found {len(paths)} ensemble files in {ensemble_dir}")
    for p in paths:
        print(f"  {os.path.basename(p)}")

    ch_idx = channel - 1

    fig, axes = plt.subplots(len(paths) + 1, 1, figsize=(14, 2 + len(paths) * 1.4), sharex=True)

    # First panel: an arbitrary "reference" (we'll just use seed 0 ourselves
    # but if you have the clean recording handy you could load it here too)
    df0, emg0, t, fs = load_recording(paths[0])

    # All seeds, each in their own row
    for i, path in enumerate(paths):
        df, emg, _, _ = load_recording(path)
        axes[i].plot(t, emg[:, ch_idx], linewidth=0.4, color=CHANNEL_COLORS[ch_idx])
        axes[i].set_ylabel(f'seed {i}', rotation=0, labelpad=22, fontsize=9)
        axes[i].axhline(0, color='gray', linewidth=0.3, linestyle='--')
        axes[i].set_ylim(-1.5, 1.5)
        axes[i].grid(alpha=0.15)

    # Last panel: all seeds overlaid in low alpha so we can see the spread
    for i, path in enumerate(paths):
        df, emg, _, _ = load_recording(path)
        axes[-1].plot(t, emg[:, ch_idx], linewidth=0.4, alpha=0.5, label=f'seed {i}')
    axes[-1].set_ylabel('overlay', rotation=0, labelpad=22, fontsize=9)
    axes[-1].axhline(0, color='gray', linewidth=0.3, linestyle='--')
    axes[-1].set_ylim(-1.5, 1.5)
    axes[-1].grid(alpha=0.15)
    axes[-1].legend(loc='upper right', fontsize=7, ncol=len(paths))
    axes[-1].set_xlabel('Time (s)')

    plt.suptitle(
        f"Multi-seed ensemble (CH{channel}): same input, same sweat level, different artifact realizations",
        y=0.995,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize SweatProcessor effects on EMG recordings."
    )
    subs = parser.add_subparsers(dest='mode', required=True)

    p_cmp = subs.add_parser('compare', help='Clean vs contaminated side by side')
    p_cmp.add_argument('clean',        help='Path to clean input CSV')
    p_cmp.add_argument('contaminated', help='Path to contaminated CSV')
    p_cmp.add_argument('--save',       help='Save figure to this path instead of showing')

    p_swp = subs.add_parser('sweep', help='Visualize a sweep recording')
    p_swp.add_argument('sweep_csv',    help='Path to sweep output CSV')
    p_swp.add_argument('--channel',    type=int, default=5)
    p_swp.add_argument('--start',      type=float, default=0.0)
    p_swp.add_argument('--end',        type=float, default=1.0)
    p_swp.add_argument('--save',       help='Save figure to this path instead of showing')

    p_ens = subs.add_parser('ensemble', help='Overlay multi-seed outputs')
    p_ens.add_argument('directory',    help='Directory containing multi-seed CSV files')
    p_ens.add_argument('--channel',    type=int, default=5)
    p_ens.add_argument('--save',       help='Save figure to this path instead of showing')

    args = parser.parse_args()

    if args.mode == 'compare':
        mode_compare(args.clean, args.contaminated, args.save)
    elif args.mode == 'sweep':
        mode_sweep(args.sweep_csv, args.channel, args.start, args.end, args.save)
    elif args.mode == 'ensemble':
        mode_ensemble(args.directory, args.channel, args.save)


if __name__ == '__main__':
    main()