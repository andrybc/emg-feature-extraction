"""
EMG Armband Simulator
=====================
A real-time 8-channel surface EMG simulator with a live oscilloscope display,
gesture controls, and CSV data output.

The output CSV is directly compatible with the notebook preprocessing pipeline
(filtering, windowing, feature extraction, classification).

Run in your .venv:
    python emg_simulator.py

Build a standalone .exe (see build_exe.bat / build_exe.sh):
    pyinstaller --onefile --windowed --name EMGSimulator emg_simulator.py

Requirements:
    pip install numpy scipy matplotlib pyinstaller

Author: EMG Research Pipeline Tutorial
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import os
import time
import threading
import queue
import csv
from datetime import datetime
from collections import deque

import numpy as np
import scipy.signal as dsp

# Matplotlib must be told to use Tkinter BEFORE any other matplotlib import.
# 'TkAgg' is the backend that draws matplotlib figures inside a Tkinter window.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

import tkinter as tk
import tempfile
import shutil
from tkinter import messagebox, filedialog
# =============================================================================
# PATH RESOLUTION
# When packaged with PyInstaller (frozen), __file__ does not exist.
# We detect whether we are running as a script or as a frozen .exe.
# This ensures the output folder is always placed next to the executable.
# =============================================================================

if getattr(sys, 'frozen', False):
    # Running as a PyInstaller .exe bundle
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as a normal .py script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'simulated')

# =============================================================================
# GLOBAL SIGNAL PARAMETERS
# These are the core parameters of the EMG pipeline.
# If you change FS here, the output CSV will match, and the notebook
# preprocessing pipeline (which also reads FS from the file) will adapt.
# =============================================================================

DEFAULT_FS       = 1000   # Default sampling rate in Hz
N_CHANNELS       = 8      # Number of EMG channels (armband electrodes)
DISPLAY_SECONDS  = 3      # How many seconds of signal to show in the plot
BATCH_SIZE       = 10
ANIM_INTERVAL_MS = 33

# Valid sample rates the user can choose from
VALID_SAMPLE_RATES = [10, 100, 250, 500, 1000, 2000, 4000, 5000]

THEMES = {
    'dark': {
        'bg_primary':     '#0f172a',
        'bg_secondary':   '#1e293b',
        'bg_plot':        '#07101f',
        'bg_plot_alt':    '#091322',
        'bg_statusbar':   '#0a1628',
        'fg_primary':     '#e2e8f0',
        'fg_secondary':   '#475569',
        'fg_dim':         '#334155',
        'fg_accent':      '#38bdf8',
        'fg_live':        '#34d399',
        'fg_rec':         '#f87171',
        'spine_color':    '#0d1e3a',
        'zero_line':      '#0f2040',
        'tick_color':     '#1e3a5f',
        'btn_bg':         '#1e293b',
        'btn_active_bg':  '#263d5e',
        'plot_line_alpha': 0.9,
    },
    'light': {
        'bg_primary':     '#f1f5f9',
        'bg_secondary':   '#e2e8f0',
        'bg_plot':        '#ffffff',
        'bg_plot_alt':    '#f8fafc',
        'bg_statusbar':   '#cbd5e1',
        'fg_primary':     '#0f172a',
        'fg_secondary':   '#475569',
        'fg_dim':         '#64748b',
        'fg_accent':      '#0284c7',
        'fg_live':        '#16a34a',
        'fg_rec':         '#dc2626',
        'spine_color':    '#cbd5e1',
        'zero_line':      '#e2e8f0',
        'tick_color':     '#94a3b8',
        'btn_bg':         '#e2e8f0',
        'btn_active_bg':  '#bfdbfe',
        'plot_line_alpha': 1.0,
    },
}
# =============================================================================
# GESTURE CONFIGURATION
# Each gesture has:
#   label     -- integer class label (what the classifier learns to predict)
#   amplitude -- peak signal amplitude during activation (arbitrary mV units)
#   color     -- color used in the UI for this gesture
#   pattern   -- 8-element array: how strongly each channel fires for this gesture
#
# RESEARCH NOTE:
# The pattern array is what makes gestures distinguishable.
# Hand Closed activates flexor muscles (channels 3-6).
# Hand Open activates extensor muscles (channels 1,2,7,8).
# A classifier learns these patterns from the feature matrix.
# =============================================================================

GESTURE_CONFIG = {
    'Rest': {
        'label':     0,
        'amplitude': 0.065,
        'color':     '#94a3b8',
        'pattern':   np.array([0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]),
    },
    'Hand Open': {
        'label':     1,
        'amplitude': 0.88,
        'color':     '#34d399',
        'pattern':   np.array([0.93, 0.87, 0.28, 0.12, 0.10, 0.22, 0.80, 0.93]),
    },
    'Hand Closed': {
        'label':     2,
        'amplitude': 1.00,
        'color':     '#f87171',
        'pattern':   np.array([0.12, 0.22, 0.90, 0.97, 0.90, 0.80, 0.18, 0.10]),
    },
    'Wrist Flexion': {
        'label':     3,
        'amplitude': 0.78,
        'color':     '#60a5fa',
        'pattern':   np.array([0.10, 0.16, 0.50, 0.90, 0.97, 0.70, 0.16, 0.10]),
    },
    'Wrist Extension': {
        'label':     4,
        'amplitude': 0.78,
        'color':     '#fbbf24',
        'pattern':   np.array([0.90, 0.97, 0.20, 0.10, 0.10, 0.16, 0.90, 0.80]),
    },
}

# Color for each of the 8 channels (used in the oscilloscope plot)
CHANNEL_COLORS = [
    '#34d399', '#38bdf8', '#fb923c', '#f472b6',
    '#a78bfa', '#fbbf24', '#4ade80', '#f87171',
]

# =============================================================================
# EMG SIGNAL GENERATOR
# Generates realistic-looking simulated EMG using:
#   1. Gaussian noise as the random source (motor unit action potentials)
#   2. An IIR filter to shape the frequency spectrum (EMG is band-limited)
#   3. A smooth activation envelope (muscles do not switch on instantly)
#   4. Per-channel activation patterns (different muscles per gesture)
# =============================================================================

class EMGSignalGenerator:
    """
    Produces multi-channel EMG samples on demand.

    Call set_gesture() to change the active gesture.
    Call generate_samples(n) to get n new samples across all channels.

    This class is not thread-safe by itself. The caller (data loop thread)
    is the only one writing to it, so no lock is needed here.
    """

    def __init__(self, fs=DEFAULT_FS, n_channels=N_CHANNELS):
        self.fs          = fs
        self.n_channels  = n_channels
        self.active_gesture = 'Rest'

        # Envelope: 0.0 = fully at rest, 1.0 = fully activated
        # Updated each sample to create a smooth rise/fall
        self.envelope = 0.0

        # Per-channel IIR filter state (x = input history, y = output history)
        # This shapes white noise into band-limited EMG-like noise.
        # The coefficients approximate a bandpass characteristic without
        # calling scipy.signal.butter in real time (which would be slow).
        self._fs_state = [
            {'x1': 0.0, 'x2': 0.0, 'y1': 0.0, 'y2': 0.0}
            for _ in range(n_channels)
        ]

    def set_gesture(self, gesture_name: str):
        """Switch the active gesture. The envelope will transition smoothly."""
        if gesture_name in GESTURE_CONFIG:
            self.active_gesture = gesture_name

    def _shape_noise(self, channel: int, amplitude: float) -> float:
        """
        Apply a simple recursive IIR filter to white noise.

        This gives the noise a more EMG-like frequency shape:
        more energy in the 50-300 Hz range, less in DC and very high frequencies.

        The formula is a simplified 2nd-order difference equation:
            y[n] = 0.62*x[n] - 0.62*x[n-2] - 0.22*y[n-1] + 0.04*y[n-2]

        This is NOT a proper Butterworth filter (that is applied in the notebook
        preprocessing step). This is just a fast approximation for visual realism.
        """
        # Draw one sample from N(0,1) -- approximated by summing 3 uniform RVs
        # (Central Limit Theorem: sum of uniforms approaches Gaussian)
        x = np.random.randn()

        f = self._fs_state[channel]
        y = 0.62*x - 0.62*f['x2'] - 0.22*f['y1'] + 0.04*f['y2']

        # Shift filter memory
        f['x2'] = f['x1']
        f['x1'] = x
        f['y2'] = f['y1']
        f['y1'] = y

        return y * amplitude * 0.38

    def generate_samples(self, n_samples: int = 1):
        """
        Generate n_samples of 8-channel simulated EMG.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate (e.g., 10 for a 10ms batch at 1000Hz)

        Returns
        -------
        samples : np.ndarray, shape (n_samples, n_channels)
            Raw simulated EMG voltages in mV
        label : int
            The integer gesture label for this batch
        """
        cfg      = GESTURE_CONFIG[self.active_gesture]
        is_rest  = (self.active_gesture == 'Rest')
        label    = cfg['label']

        # Target envelope: 0 for rest, 1 for any gesture
        # Speed controls how fast the envelope rises (gesture start) or falls (rest)
        env_target = 0.0 if is_rest else 1.0
        env_speed  = 0.04 if is_rest else 0.06   # per-sample factor

        samples = np.zeros((n_samples, self.n_channels))

        for i in range(n_samples):
            # Step the envelope toward its target (smooth exponential approach)
            self.envelope += (env_target - self.envelope) * env_speed

            # Compute per-channel amplitude
            for ch in range(self.n_channels):
                # At rest: use the Rest gesture's amplitude (noise floor)
                # During gesture: interpolate to gesture's peak amplitude * pattern weight
                base_amp = GESTURE_CONFIG['Rest']['amplitude']
                peak_amp = cfg['amplitude'] * cfg['pattern'][ch]
                amp = base_amp + (peak_amp - base_amp) * self.envelope

                samples[i, ch] = self._shape_noise(ch, amp)

        return samples, label


# =============================================================================
# DATA RECORDER
# Writes EMG samples to a CSV file in a background thread.
# Thread-safe: the data generation thread pushes to a queue,
# the writer thread drains the queue and writes to disk.
#
# Output CSV format (one row per sample):
#   timestamp_s, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, label, gesture
#
# This format is directly compatible with pandas.read_csv() in the notebook.
# =============================================================================

class DataRecorder:
    """
    Records EMG samples to a CSV file in real time.

    Usage:
        recorder.start(output_path)       # Begin recording
        recorder.push(samples, label, ..) # Call from data thread
        recorder.stop()                   # Flush and close file
    """

    def __init__(self):
        self.is_recording  = False
        self.output_path   = None
        self.sample_count  = 0
        self._file         = None
        self._writer       = None
        self._queue        = queue.Queue(maxsize=5000)  # backpressure limit
        self._writer_thread = None

    def start(self, output_path: str):
        """Begin recording to the given CSV file path."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.output_path  = output_path
        self.sample_count = 0
        self.is_recording = True

        self._file   = open(output_path, 'w', newline='', buffering=1)
        self._writer = csv.writer(self._file)

        # Write CSV header row
        # The notebook pipeline reads this with: pd.read_csv(path)
        header = (
            ['timestamp_s']
            + [f'ch{i+1}_mV' for i in range(N_CHANNELS)]
            + ['gesture_label', 'gesture_name']
        )
        self._writer.writerow(header)

        # Save a companion metadata file (human-readable info about the recording)
        meta_path = output_path.replace('.csv', '_info.txt')
        with open(meta_path, 'w') as mf:
            mf.write("EMG Simulator Recording\n")
            mf.write("=" * 40 + "\n")
            mf.write(f"Recorded:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            mf.write(f"Sampling rate: {self.fs} Hz\n")
            mf.write(f"Channels:      {N_CHANNELS}\n")
            mf.write(f"Output file:   {os.path.basename(output_path)}\n\n")
            mf.write("Gesture Labels:\n")
            for name, cfg in GESTURE_CONFIG.items():
                mf.write(f"  {cfg['label']} = {name}\n")
            mf.write("\nLoad in Python:\n")
            mf.write(f"  import pandas as pd\n")
            mf.write(f"  df = pd.read_csv('{os.path.basename(output_path)}')\n")
            mf.write(f"  emg = df[[f'ch{{i+1}}_mV' for i in range(8)]].values\n")
            mf.write(f"  labels = df['gesture_label'].values\n")

        # Start background writer thread
        self._writer_thread = threading.Thread(
            target=self._write_loop, daemon=True, name='EMGWriterThread'
        )
        self._writer_thread.start()

    def push(self, samples: np.ndarray, label: int, gesture_name: str, t_start: float):
        """
        Queue samples for writing (non-blocking, called from data thread).

        Parameters
        ----------
        samples      : np.ndarray, shape (n_samples, n_channels)
        label        : int  -- gesture integer label
        gesture_name : str  -- gesture string name
        t_start      : float -- timestamp of first sample in this batch (seconds)
        """
        if self.is_recording:
            try:
                self._queue.put_nowait((samples.copy(), label, gesture_name, t_start))
            except queue.Full:
                pass  # Drop batch if queue is full (prevents blocking the data thread)

    def stop(self):
        """Stop recording, flush queue, close file."""
        self.is_recording = False
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=3.0)
        if self._file:
            self._file.flush()
            self._file.close()
            self._file    = None
            self._writer  = None

    def _write_loop(self):
        """
        Background writer thread.
        Continuously drains the sample queue and writes to CSV.
        Exits when is_recording is False AND the queue is empty.
        """
        while self.is_recording or not self._queue.empty():
            try:
                samples, label, gesture_name, t_start = self._queue.get(timeout=0.05)

                for i, row_vals in enumerate(samples):
                    timestamp = t_start + i / self.fs
                    csv_row   = (
                        [f'{timestamp:.6f}']
                        + [f'{v:.8f}' for v in row_vals]
                        + [str(label), gesture_name]
                    )
                    self._writer.writerow(csv_row)
                    self.sample_count += 1

            except queue.Empty:
                continue  # Nothing to write yet -- loop and check again


# =============================================================================
# MAIN APPLICATION
# Combines all components into a Tkinter window with:
#   - Matplotlib oscilloscope display (8 channels, live scrolling)
#   - Gesture control buttons
#   - Record / Stop buttons
#   - Status bar
# =============================================================================

class EMGSimulatorApp:
    """
    The main application window.

    Architecture:
    - Data thread (background): generates samples at ~FS Hz, fills display buffer
    - FuncAnimation (main thread): reads buffer and redraws plot at ~30fps
    - Tkinter event loop (main thread): handles button clicks
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EMG Armband Simulator")
        self.root.configure(bg='#0f172a')
        self.root.geometry("1150x800")
        self.root.minsize(900, 680)

        # Core components
        self.generator = EMGSignalGenerator()
        self.recorder  = DataRecorder()


        self.fs              = DEFAULT_FS  # Current sampling rate (can be changed live)
        self.current_gesture = 'Rest'      # String name of active gesture
        self.is_running      = True        # Set to False on window close
        self.total_elapsed   = 0.0         # Total seconds since app started
        self.rec_start_time  = None        # When current recording started

        self.theme_name      = 'dark'
        self.theme           = THEMES['dark']

        # Channel order: channel_order[display_pos] = actual channel index
        # e.g. [0,1,2,3,4,5,6,7] = default, [2,0,1,...] = CH3 moved to top
        self.channel_order   = list(range(N_CHANNELS))

        # channel_enabled[ch_idx] = True means that channel is shown
        self.channel_enabled = [True] * N_CHANNELS

        # Drag and drop state
        self._drag_src_pos  = None   # position being dragged from
        self._drag_tgt_pos  = None   # position being hovered over
        self._drag_ghost_wn = None   # the floating ghost Toplevel window

        # ---- Display buffer ----
        # Each channel gets a deque of length DISPLAY_SAMPLES.
        # A deque with maxlen automatically drops the oldest value when
        # a new one is appended -- this is a circular (ring) buffer.
        # deque is used instead of a numpy array here because appending
        # to a deque is O(1) vs O(n) for numpy array concatenation.
        
        # Build display buffers based on current fs
        # These get rebuilt whenever the sample rate changes
        display_samples = self.fs * DISPLAY_SECONDS
        self.display_buffers = [
            deque(np.zeros(display_samples), maxlen=display_samples)
            for _ in range(N_CHANNELS)
        ]

        self.time_axis = np.linspace(-DISPLAY_SECONDS, 0, display_samples)

        # ---- Application state ----

        # Lock protects display_buffers from concurrent read/write
        # (data thread writes, animation callback reads)
        self._buffer_lock = threading.Lock()

        # Build UI, then start background processes
        self._build_ui()
        self._start_data_thread()
        self._start_animation()

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # All layout is done here. We build top-to-bottom:
    #   1. Top bar (title + status)
    #   2. Oscilloscope plot (matplotlib embedded in tkinter)
    #   3. Control panel (gesture buttons + recording)
    #   4. Status bar (sample info + file path)
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ============================================================
        # 1. TOP BAR
        # ============================================================
        top = tk.Frame(self.root, bg='#0f172a', height=48)
        top.pack(fill='x', padx=14, pady=(10, 4))
        top.pack_propagate(False)

        tk.Label(top,
                 text="EMG Armband Simulator",
                 bg='#0f172a', fg='#38bdf8',
                 font=('Helvetica', 16, 'bold')
                 ).pack(side='left', padx=(0, 10))

        self.subtitle_label = tk.Label(top,
                 text=f"8-channel · {DEFAULT_FS} Hz · Simulated surface EMG",
                 bg='#0f172a', fg='#334155',
                 font=('Helvetica', 10))
        self.subtitle_label.pack(side='left')

        self.theme_btn = tk.Button(
            top,
            text="☀",
            command=self._toggle_theme,
            bg='#1e293b', fg='#94a3b8',
            activebackground='#263d5e', activeforeground='#e2e8f0',
            relief='flat',
            font=('Helvetica', 9, 'bold'),
            cursor='hand2',
            padx=8, pady=4
        )
        self.theme_btn.pack(side='right', padx=(0, 12))

        self.live_label = tk.Label(top,
                                   text="● LIVE",
                                   bg='#0f172a', fg='#34d399',
                                   font=('Helvetica', 11, 'bold'))
        self.live_label.pack(side='right')

        self.gesture_label = tk.Label(top,
                                      text="REST",
                                      bg='#0f172a', fg='#94a3b8',
                                      font=('Courier', 12, 'bold'))
        self.gesture_label.pack(side='right', padx=20)

        # ============================================================
        # 2. OSCILLOSCOPE PLOT + CHANNEL PANEL (horizontal container)
        # ============================================================
        plot_container = tk.Frame(self.root, bg=self.theme['bg_primary'])
        plot_container.pack(fill='both', expand=True, padx=14, pady=4)

        # Channel panel goes on the RIGHT -- must be packed before plot_frame
        # because pack() assigns space in the order widgets are packed
        # If we packed plot_frame first with expand=True it would take everything
        self._build_channel_panel(plot_container)

        # Plot frame fills the remaining left space
        plot_frame = tk.Frame(plot_container, bg=self.theme['bg_plot'])
        plot_frame.pack(side='left', fill='both', expand=True)

        # Create a matplotlib figure with 8 subplots sharing the x axis
        # sharex=True: zooming or panning one plot affects all others
        self.fig, self.axes = plt.subplots(
            N_CHANNELS, 1,
            figsize=(11, 7),
            sharex=True,
            facecolor='#07101f'
        )
        # Tight subplots: reduce whitespace between panels
        self.fig.subplots_adjust(left=0.065, right=0.98, top=0.98, bottom=0.07, hspace=0.06)

        self.lines = []   # Store line objects for updating in animation

        for ch, ax in enumerate(self.axes):
            # Alternating row background colors (like a real oscilloscope)
            ax.set_facecolor('#091322' if ch % 2 == 0 else '#07101f')

            # Y axis: fixed range; EMG amplitude rarely exceeds ±1.5 in these units
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlim(-DISPLAY_SECONDS, 0)

            # Zero reference line (dashed, very faint)
            ax.axhline(0, color='#0f2040', linewidth=0.5, linestyle='--', zorder=0)

            # Spine styling (the box around each subplot)
            for spine in ax.spines.values():
                spine.set_color('#0d1e3a')
                spine.set_linewidth(0.5)

            # Channel label on y axis
            ax.set_ylabel(f'CH{ch+1}',
                          color=CHANNEL_COLORS[ch],
                          fontsize=8, fontweight='bold',
                          fontfamily='monospace',
                          rotation=0, labelpad=26)
            ax.yaxis.set_label_coords(-0.042, 0.35)

            # Minimal y tick marks (just -1, 0, 1)
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(axis='y', labelsize=5, colors='#1e3a5f')

            # Create the line object (initially all zeros)
            # We will update line.set_ydata() in the animation callback
            line, = ax.plot(
                self.time_axis,
                np.zeros(self.fs * DISPLAY_SECONDS),
                color=CHANNEL_COLORS[ch],
                linewidth=0.85,
                antialiased=True,
                zorder=2
            )
            self.lines.append(line)

        # x axis label only on the bottom subplot
        self.axes[-1].set_xlabel('Time (seconds)', color='#475569', fontsize=9)
        self.axes[-1].tick_params(axis='x', labelsize=8, colors='#475569')

        # Embed the matplotlib figure into the tkinter window
        # FigureCanvasTkAgg creates a Tkinter widget that contains the figure
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.mpl_canvas.draw()
        self.mpl_canvas.get_tk_widget().pack(fill='both', expand=True)

        # ============================================================
        # 3. CONTROL PANEL
        # ============================================================
        ctrl = tk.Frame(self.root, bg='#0f172a')
        ctrl.pack(fill='x', padx=14, pady=(4, 6))

        # -- Gesture buttons --
        gesture_frame = tk.LabelFrame(
            ctrl,
            text="  Gesture Control  ",
            bg='#0f172a', fg='#334155',
            font=('Helvetica', 9),
            bd=1, relief='groove'
        )
        gesture_frame.pack(side='left', padx=(0, 10))

        btn_row = tk.Frame(gesture_frame, bg='#0f172a')
        btn_row.pack(padx=10, pady=8)

        self._gesture_btns = {}
        for name, cfg in GESTURE_CONFIG.items():
            btn = tk.Button(
                btn_row,
                text=name,
                command=lambda g=name: self._activate_gesture(g),
                bg='#1e293b',
                fg=cfg['color'],
                activebackground='#263d5e',
                activeforeground=cfg['color'],
                relief='flat',
                font=('Helvetica', 10, 'bold'),
                width=14,
                cursor='hand2',
                padx=8, pady=7
            )
            btn.pack(side='left', padx=4)
            self._gesture_btns[name] = btn

        self._refresh_button_styles('Rest')

        # -- Recording controls --
        rec_frame = tk.LabelFrame(
            ctrl,
            text="  Data Output  ",
            bg='#0f172a', fg='#334155',
            font=('Helvetica', 9),
            bd=1, relief='groove'
        )
        rec_frame.pack(side='left', padx=(0, 10))

        rec_inner = tk.Frame(rec_frame, bg='#0f172a')
        rec_inner.pack(padx=10, pady=8)

        self.rec_btn = tk.Button(
            rec_inner,
            text="  Start Recording",
            command=self._toggle_recording,
            bg='#1e293b', fg='#f87171',
            activebackground='#263d5e', activeforeground='#f87171',
            relief='flat', font=('Helvetica', 10, 'bold'),
            width=18, cursor='hand2', padx=8, pady=7
        )
        self.rec_btn.pack(side='left', padx=(0, 10))

        self.rec_status = tk.Label(
            rec_inner,
            text="Not recording",
            bg='#0f172a', fg='#334155',
            font=('Courier', 9)
        )
        self.rec_status.pack(side='left')

        # -- Sample rate control --
        fs_frame = tk.LabelFrame(
            ctrl,
            text="  Sample Rate  ",
            bg='#0f172a', fg='#334155',
            font=('Helvetica', 9),
            bd=1, relief='groove'
        )
        fs_frame.pack(side='left', padx=(0, 10))

        fs_inner = tk.Frame(fs_frame, bg='#0f172a')
        fs_inner.pack(padx=10, pady=8)

        # StringVar holds the currently selected sample rate as a string
        # OptionMenu (tkinter dropdown) updates this variable when user picks a rate
        self._fs_var = tk.StringVar(value=str(self.fs))

        fs_label = tk.Label(
            fs_inner,
            text="Fs (Hz):",
            bg='#0f172a', fg='#475569',
            font=('Helvetica', 9)
        )
        fs_label.pack(side='left', padx=(0, 6))

        # OptionMenu: first arg = parent, second = StringVar, rest = menu options
        # The lambda traces the StringVar and calls _change_sample_rate when it changes
        fs_menu = tk.OptionMenu(
            fs_inner,
            self._fs_var,
            *[str(r) for r in VALID_SAMPLE_RATES],
            command=lambda val: self._change_sample_rate(int(val))
        )
        fs_menu.config(
            bg='#1e293b', fg='#38bdf8',
            activebackground='#263d5e', activeforeground='#38bdf8',
            highlightthickness=0,
            relief='flat',
            font=('Helvetica', 10, 'bold'),
            width=6,
            cursor='hand2'
        )
        fs_menu['menu'].config(
            bg='#1e293b', fg='#38bdf8',
            activebackground='#263d5e', activeforeground='#38bdf8',
            font=('Helvetica', 10)
        )
        fs_menu.pack(side='left')

        # -- Info panel (right side) --
        info_frame = tk.Frame(ctrl, bg='#0f172a')
        info_frame.pack(side='right', padx=8)

        self.file_info_label = tk.Label(
            info_frame,
            text=f"Output: {os.path.join('data', 'simulated', '')}",
            bg='#0f172a', fg='#1e3a5f',
            font=('Courier', 8),
            justify='right'
        )
        self.file_info_label.pack(anchor='e')

        # ============================================================
        # 4. STATUS BAR
        # ============================================================
        status_bar = tk.Frame(self.root, bg='#0a1628', height=24)
        status_bar.pack(fill='x', side='bottom')
        status_bar.pack_propagate(False)

        self.sample_rate_label = tk.Label(
            status_bar,
            text=f"Fs = {DEFAULT_FS} Hz  |  Channels = {N_CHANNELS}  |  Display window = {DISPLAY_SECONDS}s",
            bg='#0a1628', fg='#1e3a5f',
            font=('Courier', 8)
        )
        self.sample_rate_label.pack(side='left', padx=10)

        self.elapsed_lbl = tk.Label(
            status_bar,
            text="Elapsed: 0.0 s",
            bg='#0a1628', fg='#1e3a5f',
            font=('Courier', 8)
        )
        self.elapsed_lbl.pack(side='right', padx=10)



    def _build_channel_panel(self, parent):
        """
        Build the right-side channel management panel.
        Contains one draggable row per channel with an ON/OFF toggle.
        """
        t = self.theme

        # Outer panel -- fixed width, full height
        self._ch_panel = tk.Frame(
            parent,
            bg=t['bg_primary'],
            width=148
        )
        self._ch_panel.pack(side='right', fill='y', padx=(6, 0))
        self._ch_panel.pack_propagate(False)  # Prevent children from resizing it

        # Title
        tk.Label(
            self._ch_panel,
            text="Channels",
            bg=t['bg_primary'],
            fg=t['fg_accent'],
            font=('Helvetica', 9, 'bold')
        ).pack(pady=(10, 3))

        # Thin separator line
        tk.Frame(
            self._ch_panel,
            bg=t['fg_dim'],
            height=1
        ).pack(fill='x', padx=8, pady=(0, 6))

        # Inner scrollable frame for rows
        self._ch_rows_frame = tk.Frame(self._ch_panel, bg=t['bg_primary'])
        self._ch_rows_frame.pack(fill='both', expand=True, padx=6)

        # Storage for row frame references (rebuilt on each reorder)
        self._ch_row_frames = []
        self._rebuild_channel_rows()

        # Thin separator line above reset button
        tk.Frame(
            self._ch_panel,
            bg=t['fg_dim'],
            height=1
        ).pack(fill='x', padx=8, pady=(6, 4))

        # Reset button
        self._reset_btn = tk.Button(
            self._ch_panel,
            text="Reset Order",
            command=self._reset_channel_order,
            bg=t['btn_bg'],
            fg=t['fg_secondary'],
            activebackground=t['btn_active_bg'],
            activeforeground=t['fg_primary'],
            relief='flat',
            font=('Helvetica', 8),
            cursor='hand2',
            pady=5
        )
        self._reset_btn.pack(fill='x', padx=8, pady=(0, 10))


    def _rebuild_channel_rows(self):
            """
            Destroy all existing channel rows and recreate them
            in the current channel_order. Called after every reorder or toggle.
            """
            t = self.theme

            # Destroy existing rows cleanly
            for frame in self._ch_row_frames:
                frame.destroy()
            self._ch_row_frames = []

            for display_pos in range(N_CHANNELS):
                ch_idx  = self.channel_order[display_pos]
                color   = CHANNEL_COLORS[ch_idx]
                enabled = self.channel_enabled[ch_idx]

                # Row background -- slightly different from panel bg
                row_bg = t['bg_secondary']

                row = tk.Frame(
                    self._ch_rows_frame,
                    bg=row_bg,
                    relief='flat',
                    padx=3,
                    pady=4
                )
                row.pack(fill='x', pady=2)

                # Drag handle -- cursor changes to crosshair to signal draggability
                handle = tk.Label(
                    row,
                    text="⠿",
                    bg=row_bg,
                    fg=t['fg_dim'],
                    font=('Helvetica', 12),
                    cursor='fleur'
                )
                handle.pack(side='left', padx=(2, 3))

                # Colored dot showing channel identity
                dot = tk.Label(
                    row,
                    text="●",
                    bg=row_bg,
                    fg=color if enabled else t['fg_dim'],
                    font=('Helvetica', 10)
                )
                dot.pack(side='left', padx=(0, 3))

                # Channel label
                ch_lbl = tk.Label(
                    row,
                    text=f"CH{ch_idx + 1}",
                    bg=row_bg,
                    fg=color if enabled else t['fg_dim'],
                    font=('Courier', 9, 'bold'),
                    width=4,
                    anchor='w'
                )
                ch_lbl.pack(side='left')

                # ON / OFF toggle button
                en_btn = tk.Button(
                    row,
                    text="ON" if enabled else "OFF",
                    command=lambda ci=ch_idx: self._toggle_channel_enabled(ci),
                    bg='#1a3a2a' if enabled else '#3a1a1a',
                    fg='#34d399' if enabled else '#f87171',
                    activebackground=t['btn_active_bg'],
                    relief='flat',
                    font=('Courier', 7, 'bold'),
                    width=3,
                    cursor='hand2',
                    pady=2
                )
                en_btn.pack(side='right', padx=(2, 2))

                # Bind drag events to every widget in the row EXCEPT the toggle button
                # The toggle button handles its own click event
                for widget in [row, handle, dot, ch_lbl]:
                    widget.bind(
                        '<Button-1>',
                        lambda e, p=display_pos: self._drag_start(e, p)
                    )
                    widget.bind('<B1-Motion>',       self._drag_motion)
                    widget.bind('<ButtonRelease-1>', self._drag_release)

                self._ch_row_frames.append(row)


    def _rebuild_channel_rows(self):
        """
        Destroy all existing channel rows and recreate them
        in the current channel_order. Called after every reorder or toggle.
        """
        t = self.theme

        # Destroy existing rows cleanly
        for frame in self._ch_row_frames:
            frame.destroy()
        self._ch_row_frames = []

        for display_pos in range(N_CHANNELS):
            ch_idx  = self.channel_order[display_pos]
            color   = CHANNEL_COLORS[ch_idx]
            enabled = self.channel_enabled[ch_idx]

            # Row background -- slightly different from panel bg
            row_bg = t['bg_secondary']

            row = tk.Frame(
                self._ch_rows_frame,
                bg=row_bg,
                relief='flat',
                padx=3,
                pady=4
            )
            row.pack(fill='x', pady=2)

            # Drag handle -- cursor changes to crosshair to signal draggability
            handle = tk.Label(
                row,
                text="⠿",
                bg=row_bg,
                fg=t['fg_dim'],
                font=('Helvetica', 12),
                cursor='fleur'
            )
            handle.pack(side='left', padx=(2, 3))

            # Colored dot showing channel identity
            dot = tk.Label(
                row,
                text="●",
                bg=row_bg,
                fg=color if enabled else t['fg_dim'],
                font=('Helvetica', 10)
            )
            dot.pack(side='left', padx=(0, 3))

            # Channel label
            ch_lbl = tk.Label(
                row,
                text=f"CH{ch_idx + 1}",
                bg=row_bg,
                fg=color if enabled else t['fg_dim'],
                font=('Courier', 9, 'bold'),
                width=4,
                anchor='w'
            )
            ch_lbl.pack(side='left')

            # ON / OFF toggle button
            en_btn = tk.Button(
                row,
                text="ON" if enabled else "OFF",
                command=lambda ci=ch_idx: self._toggle_channel_enabled(ci),
                bg='#1a3a2a' if enabled else '#3a1a1a',
                fg='#34d399' if enabled else '#f87171',
                activebackground=t['btn_active_bg'],
                relief='flat',
                font=('Courier', 7, 'bold'),
                width=3,
                cursor='hand2',
                pady=2
            )
            en_btn.pack(side='right', padx=(2, 2))

            # Bind drag events to every widget in the row EXCEPT the toggle button
            # The toggle button handles its own click event
            for widget in [row, handle, dot, ch_lbl]:
                widget.bind(
                    '<Button-1>',
                    lambda e, p=display_pos: self._drag_start(e, p)
                )
                widget.bind('<B1-Motion>',       self._drag_motion)
                widget.bind('<ButtonRelease-1>', self._drag_release)

            self._ch_row_frames.append(row)
    


    def _drag_start(self, event, position):
        """
        User pressed mouse button on a channel row.
        Record the source position and spawn a ghost window.
        """
        self._drag_src_pos = position
        self._drag_tgt_pos = position

        ch_idx = self.channel_order[position]
        color  = CHANNEL_COLORS[ch_idx]

        # Create a small floating window that follows the cursor during drag
        # overrideredirect(True): removes all window decorations (no title bar)
        # topmost(True): always renders above other windows
        self._drag_ghost_wn = tk.Toplevel(self.root)
        self._drag_ghost_wn.overrideredirect(True)
        self._drag_ghost_wn.attributes('-alpha', 0.82)
        self._drag_ghost_wn.attributes('-topmost', True)

        tk.Label(
            self._drag_ghost_wn,
            text=f"  CH{ch_idx + 1}  ",
            bg=color,
            fg='#0f172a',
            font=('Courier', 10, 'bold'),
            padx=8,
            pady=5
        ).pack()

        # Position ghost slightly offset from the cursor so it does not
        # block the widget underneath and interfere with motion events
        self._drag_ghost_wn.geometry(
            f"+{event.x_root + 14}+{event.y_root + 6}"
        )

    def _drag_motion(self, event):
        """
        User is moving the mouse while holding the button.
        Move the ghost window and highlight the current drop target.
        """
        if self._drag_ghost_wn is None:
            return

        # Move ghost with cursor
        self._drag_ghost_wn.geometry(
            f"+{event.x_root + 14}+{event.y_root + 6}"
        )

        # Figure out which row the cursor is over
        new_tgt = self._get_drop_position(event.y_root)

        if new_tgt != self._drag_tgt_pos:
            # Remove highlight from old target row
            self._set_row_highlight(self._drag_tgt_pos, highlighted=False)

            # Apply highlight to new target row
            self._drag_tgt_pos = new_tgt
            self._set_row_highlight(self._drag_tgt_pos, highlighted=True)

    def _drag_release(self, event):
        """
        User released the mouse button.
        Perform the reorder if src != tgt, then clean up.
        """
        # Destroy ghost window
        if self._drag_ghost_wn is not None:
            self._drag_ghost_wn.destroy()
            self._drag_ghost_wn = None

        src = self._drag_src_pos
        tgt = self._get_drop_position(event.y_root)

        if src is not None and tgt is not None and src != tgt:
            # Remove the channel from its old position and insert at new position
            # list.pop(i) removes and returns element at index i
            # list.insert(i, val) inserts val before index i
            ch = self.channel_order.pop(src)
            self.channel_order.insert(tgt, ch)

            self._refresh_plot_labels()

        # Always rebuild rows to clear any highlight state
        self._rebuild_channel_rows()

        self._drag_src_pos = None
        self._drag_tgt_pos = None

    def _get_drop_position(self, y_root):
        """
        Given the mouse y position in screen coordinates,
        return which row index the cursor is closest to.

        We compare against the midpoint of each row so the
        channel snaps to a new position when you cross the halfway point.
        """
        if not self._ch_row_frames:
            return 0

        for i, row in enumerate(self._ch_row_frames):
            row_top = row.winfo_rooty()
            row_mid = row_top + row.winfo_height() // 2

            if y_root <= row_mid:
                return i

        # Mouse is below all rows -- snap to last position
        return len(self._ch_row_frames) - 1

    def _set_row_highlight(self, position, highlighted: bool):
        """
        Apply or remove a highlight color on a row and its children.
        Used to give visual feedback during drag.
        """
        if position is None or position >= len(self._ch_row_frames):
            return

        row = self._ch_row_frames[position]
        bg  = self.theme['btn_active_bg'] if highlighted else self.theme['bg_secondary']

        row.configure(bg=bg)
        for child in row.winfo_children():
            # Skip the ON/OFF button -- it has its own color logic
            if isinstance(child, tk.Button):
                continue
            try:
                child.configure(bg=bg)
            except tk.TclError:
                pass
    



    def _toggle_channel_enabled(self, ch_idx: int):
        """Toggle a single channel on or off and refresh the panel."""
        self.channel_enabled[ch_idx] = not self.channel_enabled[ch_idx]
        self._rebuild_channel_rows()

    def _reset_channel_order(self):
        """Reset to original channel order and re-enable all channels."""
        self.channel_order   = list(range(N_CHANNELS))
        self.channel_enabled = [True] * N_CHANNELS
        self._rebuild_channel_rows()
        self._refresh_plot_labels()

    def _refresh_plot_labels(self):
        """
        Update the y-axis label and line color on each subplot
        to reflect the current channel_order.

        Called after every reorder so the plot always accurately
        describes what is being displayed.
        """
        for display_pos in range(N_CHANNELS):
            ch_idx = self.channel_order[display_pos]
            color  = CHANNEL_COLORS[ch_idx]

            self.axes[display_pos].set_ylabel(
                f'CH{ch_idx + 1}',
                color=color,
                fontsize=8, fontweight='bold',
                fontfamily='monospace',
                rotation=0, labelpad=26
            )
            self.axes[display_pos].yaxis.set_label_coords(-0.042, 0.35)

            # Update the line color to match the channel now in this slot
            self.lines[display_pos].set_color(color)

        self.mpl_canvas.draw_idle()





    # ------------------------------------------------------------------
    # GESTURE ACTIVATION
    # ------------------------------------------------------------------

    def _activate_gesture(self, gesture_name: str):
        """Called when a gesture button is clicked."""
        self.current_gesture = gesture_name
        self.generator.set_gesture(gesture_name)

        # Update UI
        color = GESTURE_CONFIG[gesture_name]['color']
        self.gesture_label.config(text=gesture_name.upper(), fg=color)
        self._refresh_button_styles(gesture_name)

    def _refresh_button_styles(self, active: str):
        """Highlight the active gesture button, dim the rest."""
        for name, btn in self._gesture_btns.items():
            if name == active:
                btn.config(bg='#1e3a5f', relief='sunken')
            else:
                btn.config(bg='#1e293b', relief='flat')

    def _change_sample_rate(self, new_fs: int):
        """
        Switch the sampling rate live without restarting the app.

        Steps:
          1. Block the data thread from writing (acquire lock)
          2. Update self.fs and the generator
          3. Rebuild display buffers to match new buffer size
          4. Rebuild the time axis
          5. Update the matplotlib x-axis data on each line
          6. Release the lock so the data thread resumes at the new rate
        """
        if new_fs == self.fs:
            return   # Nothing to do

        # Block the data thread while we resize the buffers
        # Without this, the thread could write to the old buffer
        # while we are replacing it, causing a crash or corrupted display
        with self._buffer_lock:
            self.fs = new_fs
            self.generator.fs = new_fs

            new_display_samples = new_fs * DISPLAY_SECONDS

            # Rebuild each channel buffer at the new size, filled with zeros
            self.display_buffers = [
                deque(np.zeros(new_display_samples), maxlen=new_display_samples)
                for _ in range(N_CHANNELS)
            ]

            # Rebuild the time axis for the new sample count
            self.time_axis = np.linspace(-DISPLAY_SECONDS, 0, new_display_samples)

            # Update the x data on every plot line
            # The y data will be updated on the next animation frame automatically
            for line in self.lines:
                line.set_xdata(self.time_axis)
                line.set_ydata(np.zeros(new_display_samples))

        # Update the status bar to show the new rate
        self.sample_rate_label.config(
            text=f"Fs = {new_fs} Hz  |  Channels = {N_CHANNELS}  |  Display window = {DISPLAY_SECONDS}s"
        )
        self.subtitle_label.config(
            text=f"8-channel · {new_fs} Hz · Simulated surface EMG"
        )

        # Warn user if they try to record at a non-standard rate
        if new_fs not in (1000, 2000):
            self.file_info_label.config(
                text=f"Fs set to {new_fs} Hz -- note in your analysis",
                fg='#fbbf24'
            )
        else:
            self.file_info_label.config(
                text=f"Output: {os.path.join('data', 'simulated', '')}",
                fg='#1e3a5f'
            )

    def _toggle_theme(self):
            """Switch between dark and light mode."""
            if self.theme_name == 'dark':
                self.theme_name = 'light'
                self.theme      = THEMES['light']
                self.theme_btn.config(text="🌙")
            else:
                self.theme_name = 'dark'
                self.theme      = THEMES['dark']
                self.theme_btn.config(text="☀")

            self._apply_theme()

    def _apply_theme(self):
        """
        Apply the current theme to every widget and plot element.
        Called once on toggle. Works by reaching into every stored
        widget reference and updating its colors directly.
        """
        t = self.theme  # shorthand

        # ---- Root window ----
        self.root.configure(bg=t['bg_primary'])

        # ---- Top bar ----
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.configure(bg=t['bg_primary'])

        # Walk every widget in the window and recolor frames and labels
        # winfo_children() only goes one level deep, so we use a helper
        self._recolor_widgets(self.root, t)

        # ---- Plot area ----
        self.fig.set_facecolor(t['bg_plot'])

        for ch, ax in enumerate(self.axes):
            ax.set_facecolor(t['bg_plot_alt'] if ch % 2 == 0 else t['bg_plot'])

            for spine in ax.spines.values():
                spine.set_color(t['spine_color'])

            ax.tick_params(colors=t['tick_color'])
            ax.yaxis.label.set_color(CHANNEL_COLORS[ch])

            # Update zero reference line color
            for line in ax.get_lines():
                if line.get_linestyle() == '--':
                    line.set_color(t['zero_line'])

        self.axes[-1].xaxis.label.set_color(t['fg_secondary'])
        self.axes[-1].tick_params(axis='x', colors=t['fg_secondary'])

        # ---- Gesture buttons ----
        for name, btn in self._gesture_btns.items():
            cfg = GESTURE_CONFIG[name]
            if name == self.current_gesture:
                btn.configure(bg=t['btn_active_bg'], fg=cfg['color'])
            else:
                btn.configure(bg=t['btn_bg'], fg=cfg['color'],
                              activebackground=t['btn_active_bg'])

        # ---- Theme button itself ----
        self.theme_btn.configure(
            bg=t['btn_bg'],
            fg=t['fg_secondary'],
            activebackground=t['btn_active_bg']
        )

        # ---- Recording button ----
        self.rec_btn.configure(
            bg=t['btn_bg'],
            fg=t['fg_rec'],
            activebackground=t['btn_active_bg']
        )

        # ---- Status bar ----
        self.sample_rate_label.configure(
            bg=t['bg_statusbar'], fg=t['fg_dim']
        )
        self.elapsed_lbl.configure(
            bg=t['bg_statusbar'], fg=t['fg_dim']
        )

        # Clear the blit background cache so the animation picks up new colors
        # Without this, FuncAnimation keeps restoring the old cached background
        # on every frame, overwriting the theme change until a resize forces a reset

        
        # Rebuild channel rows so ON/OFF buttons and labels use new theme colors
        if hasattr(self, '_ch_rows_frame'):
            self._rebuild_channel_rows()

        # Restyle the channel panel frame and reset button
        if hasattr(self, '_ch_panel'):
            self._ch_panel.configure(bg=t['bg_primary'])
        if hasattr(self, '_reset_btn'):
            self._reset_btn.configure(
                bg=t['btn_bg'],
                fg=t['fg_secondary'],
                activebackground=t['btn_active_bg']
            )
        if hasattr(self, 'anim'):
            try:
                self.anim._blit_cache.clear()
            except AttributeError:
                pass

        # draw() forces a synchronous full redraw immediately
        # draw_idle() only schedules it, which is not enough here
        self.mpl_canvas.draw()

    def _recolor_widgets(self, parent, t):
        """
        Recursively walk all child widgets and update background
        and foreground colors to match the current theme.

        We check the widget type because different widget types
        have different config options -- Button has activebackground,
        Label does not, LabelFrame has different text color options, etc.
        """
        for widget in parent.winfo_children():
            widget_type = type(widget).__name__

            try:
                if widget_type in ('Frame', 'LabelFrame'):
                    widget.configure(bg=t['bg_primary'])
                    if widget_type == 'LabelFrame':
                        widget.configure(fg=t['fg_dim'])

                elif widget_type == 'Label':
                    widget.configure(bg=t['bg_primary'])
                    # Only update fg if it is a neutral color
                    # We do not want to overwrite gesture colors or accent colors
                    current_fg = widget.cget('fg')
                    neutral_colors = {
                        '#334155', '#475569', '#1e3a5f',
                        '#94a3b8', '#e2e8f0', '#0f172a',
                        '#64748b', '#cbd5e1'
                    }
                    if current_fg in neutral_colors:
                        widget.configure(fg=t['fg_secondary'])

                elif widget_type == 'Button':
                    # Skip gesture buttons -- handled separately above
                    pass

            except tk.TclError:
                pass  # Some widgets do not support certain config options

            # Recurse into children
            self._recolor_widgets(widget, t)
    # ------------------------------------------------------------------
    # RECORDING TOGGLE
    # ------------------------------------------------------------------

    def _toggle_recording(self):
        if not self.recorder.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        # Store gesture and timestamp so we can suggest a filename later
        self._rec_timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._rec_gesture_slug = self.current_gesture.replace(' ', '_').lower()

        # Save to a temp file while recording is active.

        fd, self._temp_csv_path = tempfile.mkstemp(suffix='.csv', prefix='emg_tmp_')
        os.close(fd)

        self.recorder.start(self._temp_csv_path)
        self.rec_start_time = time.time()

        self.rec_btn.config(text="  Stop Recording", fg='#fbbf24')
        self.live_label.config(text="● REC", fg='#f87171')
        self.rec_status.config(text="Recording...", fg='#fbbf24')
        self.file_info_label.config(text="Recording to temp file...", fg='#fbbf24')

    def _stop_recording(self):
        # Stop the recorder and flush everything to the temp file
        self.recorder.stop()

        n   = self.recorder.sample_count
        dur = n / self.fs

        # Reset UI to live state immediately
        self.rec_btn.config(text="  Start Recording", fg='#f87171')
        self.live_label.config(text="● LIVE", fg='#34d399')
        self.rec_status.config(text=f"Recorded {n:,} samples ({dur:.1f} s)", fg='#94a3b8')
        self.file_info_label.config(text="Unsaved recording in memory", fg='#94a3b8')

        # Guard: if something went wrong and we have no temp file, bail out
        if not hasattr(self, '_temp_csv_path') or not os.path.exists(self._temp_csv_path):
            messagebox.showerror("Error", "Temp recording file not found. Recording may have failed.")
            return

        # Ask the user if they want to save
        want_save = messagebox.askyesno(
            title="Save Recording?",
            message=(
                f"Recording complete.\n\n"
                f"Duration:  {dur:.2f} seconds\n"
                f"Samples:   {n:,}\n"
                f"Channels:  {N_CHANNELS}\n"
                f"Gesture:   {self._rec_gesture_slug.replace('_', ' ').title()}\n\n"
                f"Do you want to save this recording?"
            )
        )

        if not want_save:
            # User said no -- delete the temp file and clean up
            self._cleanup_temp_files()
            self.rec_status.config(text="Recording discarded", fg='#f87171')
            self.file_info_label.config(text="Not saved", fg='#475569')
            return

        # User said yes -- open a Save As dialog
        # Build a suggested filename from gesture name and timestamp
        suggested_name = f"emg_{self._rec_gesture_slug}_{self._rec_timestamp}.csv"

        # filedialog.asksaveasfilename() opens the OS native Save dialog.
        # initialdir  -- folder the dialog opens in
        # initialfile -- pre-filled filename the user can edit
        # defaultextension -- appended automatically if user omits it
        # filetypes -- the dropdown filter in the dialog
        save_path = filedialog.asksaveasfilename(
            title="Save EMG Recording",
            initialdir=os.path.expanduser("~"),   # start in home directory
            initialfile=suggested_name,
            defaultextension=".csv",
            filetypes=[
                ("CSV files",  "*.csv"),
                ("All files",  "*.*"),
            ]
        )

        # save_path is an empty string if the user clicked Cancel
        if not save_path:
            # User cancelled the Save dialog -- ask if they want to discard
            discard = messagebox.askyesno(
                title="Discard Recording?",
                message="No save location chosen. Discard this recording?"
            )
            if discard:
                self._cleanup_temp_files()
                self.rec_status.config(text="Recording discarded", fg='#f87171')
                self.file_info_label.config(text="Not saved", fg='#475569')
            else:
                # Keep temp file, let them know where it is
                self.rec_status.config(text="Kept as temp file", fg='#fbbf24')
                self.file_info_label.config(
                    text=f"Temp: {os.path.basename(self._temp_csv_path)}",
                    fg='#fbbf24'
                )
            return

        # Move the temp CSV to the chosen save location.
        # shutil.move() works across drives and filesystems, unlike os.rename().
        try:
            shutil.move(self._temp_csv_path, save_path)

            # Also move the companion _info.txt file if it exists,
            # placing it next to the saved CSV with a matching name
            temp_info = self._temp_csv_path.replace('.csv', '_info.txt')
            if os.path.exists(temp_info):
                info_save_path = save_path.replace('.csv', '_info.txt')
                shutil.move(temp_info, info_save_path)

            fname = os.path.basename(save_path)
            self.rec_status.config(text=f"Saved  {n:,} samples ({dur:.1f} s)", fg='#34d399')
            self.file_info_label.config(text=f"Saved: {fname}", fg='#34d399')

            # Confirm with load instructions
            messagebox.showinfo(
                "Saved",
                f"Recording saved successfully.\n\n"
                f"File:     {save_path}\n"
                f"Samples:  {n:,}\n"
                f"Duration: {dur:.2f} s\n\n"
                f"Load in Python:\n"
                f"  import pandas as pd\n"
                f"  df = pd.read_csv(r'{save_path}')\n"
                f"  emg = df[[f'ch{{i+1}}_mV' for i in range(8)]].values"
            )

        except Exception as e:
            messagebox.showerror(
                "Save Failed",
                f"Could not save to that location.\n\nError: {e}\n\n"
                f"Your recording is still in the temp file:\n{self._temp_csv_path}"
            )


    def _cleanup_temp_files(self):
        """Delete the temporary recording files if they exist."""
        if hasattr(self, '_temp_csv_path'):
            # Remove the CSV
            if os.path.exists(self._temp_csv_path):
                os.remove(self._temp_csv_path)

            # Remove the companion info txt if it exists
            temp_info = self._temp_csv_path.replace('.csv', '_info.txt')
            if os.path.exists(temp_info):
                os.remove(temp_info)
    # ------------------------------------------------------------------
    # BACKGROUND DATA GENERATION THREAD
    # This thread runs continuously in the background.
    # It generates EMG samples at BATCH_SIZE / FS seconds per cycle,
    # sleeping the remainder of the interval to maintain timing.
    # ------------------------------------------------------------------

    def _start_data_thread(self):
        t = threading.Thread(
            target=self._data_loop,
            daemon=True,       # daemon=True: thread dies when main window closes
            name='EMGDataThread'
        )
        t.start()

    def _data_loop(self):
        """
        Background thread: generate EMG samples at approximately FS Hz.

        Each cycle:
          1. Record the start time
          2. Generate BATCH_SIZE samples
          3. Write to display buffer (thread-safe via lock)
          4. Push to recorder queue if recording
          5. Sleep for the remaining time in the batch interval
        """
        t_elapsed = 0.0

        while self.is_running:
            t_cycle_start = time.perf_counter()

            # Generate new samples
            samples, label = self.generator.generate_samples(n_samples=BATCH_SIZE)
            gesture_name   = self.current_gesture

            # Push to display buffer (shared with animation callback)
            with self._buffer_lock:
                for ch in range(N_CHANNELS):
                    # deque.extend() appends multiple values in order
                    # Old values are automatically dropped (maxlen enforced)
                    self.display_buffers[ch].extend(samples[:, ch])

            # Push to recorder if active
            if self.recorder.is_recording:
                self.recorder.push(samples, label, gesture_name, t_elapsed)

            # Recalculate interval each cycle so rate changes take effect immediately
            batch_interval     = BATCH_SIZE / self.fs
            t_elapsed         += batch_interval
            self.total_elapsed = t_elapsed

            t_used  = time.perf_counter() - t_cycle_start
            t_sleep = batch_interval - t_used
            if t_sleep > 0:
                time.sleep(t_sleep)

    # ------------------------------------------------------------------
    # MATPLOTLIB ANIMATION
    # FuncAnimation calls _update_frame every ANIM_INTERVAL_MS milliseconds.
    # blit=True means only the changed artists (our 8 lines) are redrawn,
    # not the entire figure. This is much faster for real-time plotting.
    # ------------------------------------------------------------------

    def _start_animation(self):
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=ANIM_INTERVAL_MS,
            blit=True,
            cache_frame_data=False   # Do not cache: each frame is different
        )

    def _update_frame(self, frame_number):
        """
        Called by FuncAnimation ~30 times per second.
        Reads the latest data from display_buffers and updates the 8 plot lines.

        Returns a list of the changed artists -- required by blit=True.
        """
        with self._buffer_lock:
                    for display_pos in range(N_CHANNELS):
                        ch_idx = self.channel_order[display_pos]

                        if self.channel_enabled[ch_idx]:
                            data = np.array(self.display_buffers[ch_idx])
                        else:
                            # Disabled channel shows a flat zero line
                            data = np.zeros(len(self.time_axis))

                        self.lines[display_pos].set_ydata(data)

        # Update status labels (these are cheap Tkinter text changes)
        self.elapsed_lbl.config(text=f"Elapsed: {self.total_elapsed:.1f} s")

        if self.recorder.is_recording:
            n   = self.recorder.sample_count
            dur = n / self.fs
            self.rec_status.config(text=f"Recording...  {n:,} samples  ({dur:.1f} s)")

        # Return all line objects -- FuncAnimation will redraw only these
        return self.lines

    # ------------------------------------------------------------------
    # CLEANUP ON WINDOW CLOSE
    # ------------------------------------------------------------------

    def on_close(self):
        """Called when the user closes the window."""
        self.is_running = False

        if self.recorder.is_recording:
            self.recorder.stop()

        plt.close('all')
        self.root.destroy()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    root = tk.Tk()
    app  = EMGSimulatorApp(root)

    # Register the close handler
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Start the Tkinter event loop (blocks here until window is closed)
    root.mainloop()


if __name__ == '__main__':
    main()