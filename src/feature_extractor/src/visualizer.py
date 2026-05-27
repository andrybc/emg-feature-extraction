"""
visualizer.py
=============
Animated visualization of the EMG feature extraction process.

What this module does:
  Opens a separate Toplevel window with three coordinated panels:
    1. Raw signal of the selected channel, with a sliding highlight
       rectangle showing the current window position
    2. Feature trajectory plot: z-scored feature values over windows
       for the selected channel (4 lines for Path A, 7 for Path B)
    3. 2x2 grid of scatter plots showing feature pairings, with dots
       colored by gesture so you can see class separability emerge as
       windows accumulate

How the animation works:
  Features are extracted once (cached), then a tkinter `after()` loop
  drives frame-by-frame rendering. The user controls playback via:
    - Play/Pause button
    - Step Forward button (advance by 1 window)
    - Reset button (jump to frame 0)
    - Speed dropdown (0.1x to 8x)
    - Scrub bar (drag to any frame)

Coloring convention:
  - 5 gestures get 5 Wong-palette colors (colorblind-friendly), used
    consistently in all scatter plots and the legend.
  - 4 or 7 features get distinct Set1-palette colors in the trajectory
    plot. These intentionally differ from the gesture palette so the
    two color schemes don't get confused.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

# Use the OO matplotlib interface (Figure, not pyplot) for clean
# embedding in tkinter. Avoid pyplot state leaks across windows.
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from .pipeline import (
    extract_features_path_a,
    extract_features_path_b,
    extract_features_both,
)
from .data_loading import assign_window_labels


# ---------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------
# Wong (2011) colorblind-friendly palette for gestures. Skipping pure
# yellow (low contrast on white) and black (looks like axes).
GESTURE_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#0072B2",  # blue (6th gesture if ever needed)
    "#F0E442",  # yellow
    "#000000",  # black
]

# Set1 (Brewer) for trajectory feature lines. Chosen to NOT clash with
# the gesture palette above.
FEATURE_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange (used only for Path B/Both)
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
    "#dede00",  # dark yellow
    "#17becf",  # teal
    "#bcbd22",  # olive
]


# ---------------------------------------------------------------------
# Scatter plot pairings per path
# ---------------------------------------------------------------------
# Each tuple is (x_feature_short_name, y_feature_short_name).
# Pairings are chosen to show research-relevant feature relationships:
#   - MAV vs WL: energy vs complexity. Often the two most separable features.
#   - MAV vs ZC: energy vs frequency. The two most COMPLEMENTARY feature types.
#   - ZC vs SSC: should show high redundancy (Phinyomark's finding).
#   - WL vs SSC: complexity vs frequency.
SCATTER_PAIRINGS_A = [
    ("mav", "wl"),
    ("mav", "zc"),
    ("zc", "ssc"),
    ("wl", "ssc"),
]

# Path B pairings emphasize the Phinyomark feature set.
SCATTER_PAIRINGS_B = [
    ("mav", "wl"),
    ("mav", "wamp"),
    ("wamp", "ar1"),
    ("ar1", "ar2"),
]

# How many features each path produces per channel. Used for indexing.
FEATURES_PER_CHANNEL = {
    "A": 4,    # MAV, WL, ZC, SSC
    "B": 7,    # MAV, WL, WAMP, AR1, AR2, AR3, AR4
    "BOTH": 11,  # 4 (Path A) + 7 (Path B)
}


# ---------------------------------------------------------------------
# Main visualizer window class
# ---------------------------------------------------------------------

class FeatureVisualizerWindow:
    """
    Animated feature visualization launched from the main app.

    Receives the loaded raw data and windowing parameters; extracts
    features internally so it can switch between paths on demand.
    """

    def __init__(self, parent, loaded_data, window_size_samples,
                 step_size_samples, fs, initial_path="A",
                 zc_threshold=1e-5, ssc_threshold=1e-5, wamp_threshold=1e-5,
                 label_mode="majority"):
        """
        Parameters
        ----------
        parent : tk.Tk or tk.Toplevel
            The parent window (the main app).
        loaded_data : dict
            Result from load_emg_csv: has 'signal', 'channels', 'gestures', etc.
        window_size_samples, step_size_samples : int
            Windowing config in samples.
        fs : int
            Sample rate in Hz.
        initial_path : str
            'A', 'B', or 'BOTH'. Matches the main app's selection.
        """
        # --- Store all inputs FIRST, before any method calls. ---
        # (Lesson learned from the simulator: initialization order matters.
        # Methods called later in __init__ will reference these attrs.)
        self.parent = parent
        self.loaded_data = loaded_data
        self.window_size = window_size_samples
        self.step_size = step_size_samples
        self.fs = fs
        self.zc_threshold = zc_threshold
        self.ssc_threshold = ssc_threshold
        self.wamp_threshold = wamp_threshold
        self.label_mode = label_mode

        # Convenience handles into loaded_data.
        self.signal = loaded_data["signal"]              # (N, n_channels)
        self.channel_names = loaded_data["channels"]
        self.n_channels = len(self.channel_names)
        self.gestures_per_sample = loaded_data["gestures"]  # may be None

        # Animation state.
        self.current_frame = 0
        self.is_playing = False
        self.speed = 1.0
        self.base_interval_ms = 50  # ms per frame at 1x speed
        self._after_id = None       # handle for tkinter's after() callback
        self._suppress_scrub_callback = False  # prevents feedback loop

        # Feature extraction cache: {path_name: extraction_result_dict}.
        # Avoids re-extracting when the user toggles paths back and forth.
        self._cache = {}

        # Selected dropdowns. Path comes from the main app; channel
        # defaults to the first channel.
        self.path = initial_path
        self.channel_idx = 0

        # Placeholders for matplotlib artists. Filled in by _init_artists.
        self.fig = None
        self.canvas = None
        self.ax_signal = None
        self.ax_traj = None
        self.scatter_axes = []
        self.highlight_rect = None      # the moving window box on raw signal
        self.traj_lines = []            # one Line2D per feature
        self.scatter_collections = []   # one PathCollection per scatter axis

        # --- Now build the window. ---
        self.window = tk.Toplevel(parent)
        self.window.title("EMG Feature Visualizer")
        self.window.geometry("1280x900")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_top_controls()
        self._build_figure()
        self._build_bottom_controls()

        # Extract features for the initial path and initialize the artists.
        # If Path B is the initial choice and it's disabled, we'll fall
        # back to Path A and tell the user.
        if not self._load_or_extract(self.path):
            self.path = "A"
            self.path_var.set("A")
            self._load_or_extract(self.path)

        self._init_artists_for_current_path()
        self._render_frame(0)

    # =================================================================
    # UI construction
    # =================================================================

    def _build_top_controls(self):
        """Top bar: channel selector, path selector, gesture legend."""
        bar = ttk.Frame(self.window, padding=8)
        bar.pack(side=tk.TOP, fill=tk.X)

        # --- Channel dropdown ---
        ttk.Label(bar, text="Channel:").pack(side=tk.LEFT, padx=(0, 4))
        self.channel_var = tk.StringVar(value=self.channel_names[0])
        self.channel_dropdown = ttk.Combobox(
            bar, textvariable=self.channel_var,
            values=self.channel_names, state="readonly", width=12,
        )
        self.channel_dropdown.pack(side=tk.LEFT, padx=(0, 16))
        # <<ComboboxSelected>> fires on dropdown choice change.
        self.channel_dropdown.bind("<<ComboboxSelected>>", self._on_channel_change)

        # --- Path dropdown ---
        ttk.Label(bar, text="Feature Path:").pack(side=tk.LEFT, padx=(0, 4))
        self.path_var = tk.StringVar(value=self.path)
        path_dropdown = ttk.Combobox(
            bar, textvariable=self.path_var,
            values=["A", "B", "BOTH"], state="readonly", width=6,
        )
        path_dropdown.pack(side=tk.LEFT, padx=(0, 16))
        path_dropdown.bind("<<ComboboxSelected>>", self._on_path_change)

        # --- Gesture legend (computed from actual gestures in the data) ---
        # Inline colored squares with gesture names.
        self.gesture_legend_frame = ttk.Frame(bar)
        self.gesture_legend_frame.pack(side=tk.LEFT, padx=(16, 0))
        # Populated in _init_artists_for_current_path once we know the
        # gestures present in this extraction.

    def _build_figure(self):
        """The big matplotlib figure with all the plots."""
        # Figure size in inches: 12.8 wide x 7.5 tall at 100 dpi
        # = ~1280 x 750 px, fits nicely under the top control bar.
        self.fig = Figure(figsize=(12.8, 7.5), dpi=100,
                          facecolor="white", layout="constrained")

        # GridSpec with 4 rows:
        #   row 0: raw signal (small)
        #   row 1: trajectory plot (medium)
        #   rows 2-3: 2x2 scatter grid (large)
        gs = self.fig.add_gridspec(
            4, 2,
            height_ratios=[1.0, 1.5, 2.0, 2.0],
            hspace=0.4, wspace=0.25,
        )

        # Raw signal spans both columns.
        self.ax_signal = self.fig.add_subplot(gs[0, :])
        # Trajectory plot spans both columns.
        self.ax_traj = self.fig.add_subplot(gs[1, :])
        # Four scatter axes in a 2x2 grid.
        self.scatter_axes = [
            self.fig.add_subplot(gs[2, 0]),
            self.fig.add_subplot(gs[2, 1]),
            self.fig.add_subplot(gs[3, 0]),
            self.fig.add_subplot(gs[3, 1]),
        ]

        # Embed in tkinter.
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_bottom_controls(self):
        """Bottom bar: play/pause, step, reset, speed, scrub."""
        bar = ttk.Frame(self.window, padding=8)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Play/Pause button. Single button that toggles state and label.
        self.play_button = ttk.Button(bar, text="▶ Play",
                                      command=self._toggle_play_pause)
        self.play_button.pack(side=tk.LEFT, padx=(0, 4))

        # Step forward by one window.
        self.step_button = ttk.Button(bar, text="Step ▶|",
                                      command=self._step_forward)
        self.step_button.pack(side=tk.LEFT, padx=(0, 4))

        # Reset to frame 0.
        self.reset_button = ttk.Button(bar, text="⟲ Reset",
                                       command=self._reset)
        self.reset_button.pack(side=tk.LEFT, padx=(0, 16))

        # Speed dropdown. Combobox is cleaner than a log-scale slider.
        ttk.Label(bar, text="Speed:").pack(side=tk.LEFT, padx=(0, 4))
        self.speed_var = tk.StringVar(value="1x")
        speed_dropdown = ttk.Combobox(
            bar, textvariable=self.speed_var,
            values=["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "8x"],
            state="readonly", width=6,
        )
        speed_dropdown.pack(side=tk.LEFT, padx=(0, 16))
        speed_dropdown.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Scrub bar. This is the live frame indicator + manual jump.
        # We need to know n_windows to set its range, so this gets
        # configured properly inside _init_artists_for_current_path.
        # For now create with a placeholder range that we'll update.
        ttk.Label(bar, text="Window:").pack(side=tk.LEFT, padx=(0, 4))
        self.scrub_var = tk.IntVar(value=0)
        self.scrub_scale = ttk.Scale(
            bar, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.scrub_var, command=self._on_scrub,
            length=400,
        )
        self.scrub_scale.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

        # Frame counter label, updated each render.
        self.frame_label_var = tk.StringVar(value="0 / 0")
        ttk.Label(bar, textvariable=self.frame_label_var,
                  width=14).pack(side=tk.LEFT)

    # =================================================================
    # Feature extraction & caching
    # =================================================================

    def _load_or_extract(self, path):
        """
        Get feature data for `path`, either from cache or by extracting.

        Returns True on success, False on failure (Path B disabled).
        Stores the result in self._current_data on success.
        """
        if path in self._cache:
            self._current_data = self._cache[path]
            return True

        try:
            # Dispatch to the right extraction function.
            if path == "A":
                fm, names = extract_features_path_a(
                    self.signal, self.window_size, self.step_size,
                    zc_threshold=self.zc_threshold,
                    ssc_threshold=self.ssc_threshold,
                    channel_names=self.channel_names,
                )
            elif path == "B":
                fm, names = extract_features_path_b(
                    self.signal, self.window_size, self.step_size,
                    wamp_threshold=self.wamp_threshold,
                    channel_names=self.channel_names,
                )
            else:  # "BOTH"
                fm, names = extract_features_both(
                    self.signal, self.window_size, self.step_size,
                    zc_threshold=self.zc_threshold,
                    ssc_threshold=self.ssc_threshold,
                    wamp_threshold=self.wamp_threshold,
                    channel_names=self.channel_names,
                )
        except NotImplementedError as e:
            messagebox.showwarning(
                "Path B is disabled",
                f"{e}\n\nFalling back to Path A."
            )
            return False

        # Compute window labels if we have per-sample gestures.
        if self.gestures_per_sample is not None:
            window_labels = assign_window_labels(
                self.gestures_per_sample,
                self.window_size, self.step_size,
                mode=self.label_mode,
            )
        else:
            # No labels available: synthesize a single dummy gesture
            # so coloring still works (all dots one color).
            window_labels = np.array(["unlabeled"] * fm.shape[0], dtype=object)

        # Pre-compute z-scored trajectory data so animation is fast.
        # axis=0 means "standardize each column independently."
        means = fm.mean(axis=0)
        stds = fm.std(axis=0)
        stds[stds == 0] = 1.0  # avoid divide-by-zero for constant features
        fm_z = (fm - means) / stds

        # Pre-compute a (channel_idx, short_name) -> column lookup.
        # For path "A" or "B" this is straightforward. For "BOTH" the
        # names have an "A_" or "B_" prefix that we strip.
        n_per_channel = FEATURES_PER_CHANNEL[path]
        feature_index = self._build_feature_index(names, n_per_channel, path)

        result = {
            "feature_matrix": fm,
            "feature_matrix_zscored": fm_z,
            "feature_names": names,
            "window_labels": window_labels,
            "feature_index": feature_index,
            "n_windows": fm.shape[0],
            "n_per_channel": n_per_channel,
        }
        self._cache[path] = result
        self._current_data = result
        return True

    def _build_feature_index(self, feature_names, n_per_channel, path):
        """
        Build {(channel_idx, short_feature_name): column_idx}.

        feature_names look like 'ch1_mV_mav' (Path A or B alone) or
        'A_ch1_mV_mav' (Both mode). The short name is always the last
        underscore-separated segment.
        """
        index = {}

        if path == "BOTH":
            # Two halves: Path A columns first, Path B columns second.
            # We index only the Path A side (which is what scatter pairings
            # for "BOTH" mode use). Columns 0..(4*n_channels-1) are Path A.
            for col_idx in range(4 * self.n_channels):
                name = feature_names[col_idx]
                # Drop the 'A_' prefix.
                stripped = name[2:] if name.startswith("A_") else name
                short = stripped.rsplit("_", 1)[-1]
                ch_idx = col_idx // 4   # 4 features per channel in Path A side
                index[(ch_idx, short)] = col_idx
        else:
            for col_idx, name in enumerate(feature_names):
                short = name.rsplit("_", 1)[-1]
                ch_idx = col_idx // n_per_channel
                index[(ch_idx, short)] = col_idx

        return index

    # =================================================================
    # Artist initialization (called when path changes or on first open)
    # =================================================================

    def _init_artists_for_current_path(self):
        """
        Wipe and rebuild all matplotlib artists for the current path.

        Called on first open and whenever the path dropdown changes.
        """
        data = self._current_data
        n_windows = data["n_windows"]

        # Update scrub bar range to match the new n_windows.
        self.scrub_scale.configure(to=max(0, n_windows - 1))
        self.scrub_var.set(0)
        self.frame_label_var.set(f"0 / {n_windows - 1}")

        # Reset animation state.
        self.current_frame = 0
        self.is_playing = False
        self.play_button.configure(text="▶ Play")
        if self._after_id is not None:
            self.window.after_cancel(self._after_id)
            self._after_id = None

        # Map gesture labels to colors.
        # Sort gestures alphabetically for stable color assignment across
        # path switches (otherwise the color a gesture maps to could
        # change when re-extracting).
        unique_gestures = sorted(set(data["window_labels"]))
        self.gesture_to_color = {
            g: GESTURE_COLORS[i % len(GESTURE_COLORS)]
            for i, g in enumerate(unique_gestures)
        }
        # Pre-compute color array: one color per window.
        self.window_colors = np.array(
            [self.gesture_to_color[g] for g in data["window_labels"]]
        )

        # Rebuild the gesture legend (clear and re-add colored squares).
        self._rebuild_gesture_legend(unique_gestures)

        # --- Raw signal panel ---
        self._build_signal_panel()

        # --- Trajectory panel ---
        self._build_trajectory_panel()

        # --- Scatter panels ---
        self._build_scatter_panels()

        self.canvas.draw_idle()

    def _rebuild_gesture_legend(self, unique_gestures):
        """Refresh the colored-square legend in the top control bar."""
        # Remove old legend widgets.
        for child in self.gesture_legend_frame.winfo_children():
            child.destroy()

        ttk.Label(self.gesture_legend_frame, text="Gestures:").pack(side=tk.LEFT)
        for g in unique_gestures:
            color = self.gesture_to_color[g]
            # tk.Frame with a fixed-size colored background acts as a swatch.
            swatch = tk.Frame(self.gesture_legend_frame, width=14, height=14,
                              bg=color, highlightthickness=1,
                              highlightbackground="#444444")
            swatch.pack(side=tk.LEFT, padx=(8, 2))
            ttk.Label(self.gesture_legend_frame, text=str(g)).pack(side=tk.LEFT)

    def _build_signal_panel(self):
        """Top panel: raw signal of the selected channel + moving window box."""
        ax = self.ax_signal
        ax.clear()

        # x-axis: time in seconds.
        t = np.arange(self.signal.shape[0]) / self.fs
        ch_signal = self.signal[:, self.channel_idx]

        ax.plot(t, ch_signal, color="#444444", linewidth=0.5)
        ax.set_xlim(t[0], t[-1])

        # Symmetric y limits with a little padding above the max swing.
        ymax = float(np.max(np.abs(ch_signal))) * 1.1 + 1e-6
        ax.set_ylim(-ymax, ymax)

        ax.set_title(
            f"Raw Signal — {self.channel_names[self.channel_idx]} "
            f"(sliding window highlighted)",
            fontsize=10,
        )
        ax.set_ylabel("Amplitude (mV)", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.tick_params(labelsize=8)

        # The highlight rectangle. Starts at window 0 position.
        # Width in seconds = window_size / fs.
        w_sec = self.window_size / self.fs
        self.highlight_rect = Rectangle(
            (0, -ymax), w_sec, 2 * ymax,
            facecolor="#FFD60055", edgecolor="#FFA500", linewidth=1.5,
            zorder=10,
        )
        ax.add_patch(self.highlight_rect)

    def _build_trajectory_panel(self):
        """Middle panel: z-scored feature trajectories for the selected channel."""
        ax = self.ax_traj
        ax.clear()

        data = self._current_data
        n_windows = data["n_windows"]

        # Figure out which feature short-names exist for this channel.
        # For Path A: mav, wl, zc, ssc
        # For Path B: mav, wl, wamp, ar1, ar2, ar3, ar4
        # For Both:   shows Path A side only (the indexing we built earlier
        #             only covered Path A columns).
        # We discover the available short names by looking at the index.
        ch_features = []
        for (ch_idx, short), col_idx in data["feature_index"].items():
            if ch_idx == self.channel_idx:
                ch_features.append((short, col_idx))
        # Sort for stable line ordering (mav, wl, zc, ssc, ...).
        # Use a canonical ordering for known features; unknown ones at the end.
        canonical_order = ["mav", "wl", "zc", "ssc", "wamp",
                            "ar1", "ar2", "ar3", "ar4"]
        def sort_key(item):
            short, _ = item
            if short in canonical_order:
                return canonical_order.index(short)
            return len(canonical_order)
        ch_features.sort(key=sort_key)

        # Create one Line2D per feature, all empty initially.
        # We'll grow the data via set_data() during animation.
        self.traj_lines = []
        for i, (short, col_idx) in enumerate(ch_features):
            color = FEATURE_COLORS[i % len(FEATURE_COLORS)]
            line, = ax.plot(
                [], [],
                color=color, linewidth=1.5,
                label=short.upper(),
            )
            # Stash the column index on the line so we can fetch its data later.
            line._col_idx = col_idx
            self.traj_lines.append(line)

        # Set axis limits to fit the full extracted data range.
        # x: 0 to n_windows-1
        # y: full range of z-scored features for this channel
        if n_windows > 0:
            ax.set_xlim(0, n_windows - 1)
            # y range from the columns we'll be plotting
            cols = [line._col_idx for line in self.traj_lines]
            if cols:
                ch_z = data["feature_matrix_zscored"][:, cols]
                ymin, ymax = ch_z.min(), ch_z.max()
                pad = (ymax - ymin) * 0.1 + 1e-6
                ax.set_ylim(ymin - pad, ymax + pad)

        ax.set_title(
            f"Feature Trajectories (z-scored) — {self.channel_names[self.channel_idx]}",
            fontsize=10,
        )
        ax.set_xlabel("Window index", fontsize=9)
        ax.set_ylabel("Z-scored value", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=len(self.traj_lines))

    def _build_scatter_panels(self):
        """Bottom: 2x2 scatter grid with feature pairings, dots by gesture."""
        data = self._current_data

        # Pick pairings based on path. BOTH uses Path A pairings (cleaner).
        if self.path == "B":
            pairings = SCATTER_PAIRINGS_B
        else:
            pairings = SCATTER_PAIRINGS_A

        # Fetch all (x_arr, y_arr) data for each pairing so we can set
        # the axis limits up front (so they don't change during animation).
        self.scatter_collections = []
        self.scatter_pairings = pairings  # remember for render time
        self.scatter_data_xy = []         # list of (x_arr, y_arr) per pairing

        for ax, (fx, fy) in zip(self.scatter_axes, pairings):
            ax.clear()
            col_x = data["feature_index"].get((self.channel_idx, fx))
            col_y = data["feature_index"].get((self.channel_idx, fy))

            if col_x is None or col_y is None:
                # The requested feature isn't in the current path. Show
                # an empty axis with a hint and skip data collection.
                ax.text(0.5, 0.5,
                        f"{fx.upper()} or {fy.upper()}\nnot available for path {self.path}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="#888888")
                ax.set_xticks([])
                ax.set_yticks([])
                self.scatter_data_xy.append((None, None))
                # Append a placeholder collection so indices line up.
                coll = ax.scatter([], [], s=0)
                self.scatter_collections.append(coll)
                continue

            x_arr = data["feature_matrix"][:, col_x]
            y_arr = data["feature_matrix"][:, col_y]
            self.scatter_data_xy.append((x_arr, y_arr))

            # Set axis limits from full data range so the view doesn't
            # jump as new points accumulate.
            xpad = (x_arr.max() - x_arr.min()) * 0.05 + 1e-12
            ypad = (y_arr.max() - y_arr.min()) * 0.05 + 1e-12
            ax.set_xlim(x_arr.min() - xpad, x_arr.max() + xpad)
            ax.set_ylim(y_arr.min() - ypad, y_arr.max() + ypad)

            # Create the empty PathCollection. We'll set_offsets() each frame
            # to control how many points are visible. set_facecolor with the
            # color array assigns the per-point color.
            coll = ax.scatter([], [], s=24, alpha=0.7, edgecolors="none")
            self.scatter_collections.append(coll)

            ax.set_xlabel(f"{fx.upper()}", fontsize=9)
            ax.set_ylabel(f"{fy.upper()}", fontsize=9)
            ax.set_title(
                f"{fx.upper()} vs {fy.upper()} — {self.channel_names[self.channel_idx]}",
                fontsize=10,
            )
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

    # =================================================================
    # Rendering one frame
    # =================================================================

    def _render_frame(self, k):
        """
        Render frame k: update all artists to show data through window k.

        This is the heart of the animation. Called by playback, scrub,
        step, and reset. Safe to call with any k in [0, n_windows-1].
        """
        data = self._current_data
        n_windows = data["n_windows"]
        k = max(0, min(k, n_windows - 1))
        self.current_frame = k

        # --- 1. Move the highlight rectangle on the raw signal panel. ---
        # The window covers samples [k*step_size, k*step_size + window_size).
        # In seconds: [k*step/fs, (k*step + W)/fs).
        x_start_sec = (k * self.step_size) / self.fs
        self.highlight_rect.set_x(x_start_sec)

        # --- 2. Update trajectory lines: grow each line by one point. ---
        # Each line's x-data is [0, 1, ..., k], y-data is its column [0..k].
        x_range = np.arange(k + 1)
        for line in self.traj_lines:
            col = line._col_idx
            y_vals = data["feature_matrix_zscored"][:k + 1, col]
            line.set_data(x_range, y_vals)

        # --- 3. Update scatter plots: show first k+1 dots. ---
        for coll, (x_arr, y_arr) in zip(self.scatter_collections,
                                         self.scatter_data_xy):
            if x_arr is None:
                continue  # placeholder for unavailable features
            offsets = np.column_stack([x_arr[:k + 1], y_arr[:k + 1]])
            coll.set_offsets(offsets)
            # Per-point color from gesture mapping.
            coll.set_facecolor(self.window_colors[:k + 1])

        # --- 4. Update the scrub bar position WITHOUT firing its callback. ---
        # We set a flag to suppress the callback's effect during this update,
        # otherwise scrubbing and playback fight each other.
        self._suppress_scrub_callback = True
        self.scrub_var.set(k)
        self._suppress_scrub_callback = False

        # --- 5. Update the frame counter label. ---
        self.frame_label_var.set(f"{k} / {n_windows - 1}")

        # --- 6. Schedule a canvas redraw. ---
        # draw_idle is safer than draw in an animation loop: it coalesces
        # multiple redraws if they arrive faster than the screen refreshes.
        self.canvas.draw_idle()

    # =================================================================
    # Playback control event handlers
    # =================================================================

    def _toggle_play_pause(self):
        """Toggle between playing and paused."""
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        # If we're at the end, restart from the beginning.
        if self.current_frame >= self._current_data["n_windows"] - 1:
            self.current_frame = 0
            self._render_frame(0)
        self.is_playing = True
        self.play_button.configure(text="❚❚ Pause")
        self._schedule_next_frame()

    def _pause(self):
        self.is_playing = False
        self.play_button.configure(text="▶ Play")
        if self._after_id is not None:
            self.window.after_cancel(self._after_id)
            self._after_id = None

    def _schedule_next_frame(self):
        """Use tkinter's after() to advance the animation by one frame."""
        if not self.is_playing:
            return

        # Compute interval based on speed. Clamp to a minimum so we
        # don't try to render faster than matplotlib can keep up.
        interval = max(10, int(self.base_interval_ms / self.speed))
        self._after_id = self.window.after(interval, self._advance_frame)

    def _advance_frame(self):
        """Called by the after() timer to advance one frame."""
        if not self.is_playing:
            return

        n_windows = self._current_data["n_windows"]
        if self.current_frame >= n_windows - 1:
            # Reached the end. Stop and reset the play button.
            self._pause()
            return

        self._render_frame(self.current_frame + 1)
        self._schedule_next_frame()

    def _step_forward(self):
        """Advance exactly one frame (used while paused for careful study)."""
        # If playing, pause first.
        if self.is_playing:
            self._pause()
        n_windows = self._current_data["n_windows"]
        if self.current_frame < n_windows - 1:
            self._render_frame(self.current_frame + 1)

    def _reset(self):
        """Jump back to frame 0 and pause."""
        self._pause()
        self._render_frame(0)

    def _on_scrub(self, value):
        """User dragged the scrub bar. Jump to that frame."""
        # Ignore callbacks that we triggered ourselves during rendering.
        if self._suppress_scrub_callback:
            return
        # If playing, pause (user wants manual control).
        if self.is_playing:
            self._pause()
        # tkinter's Scale passes the new value as a string.
        new_frame = int(float(value))
        if new_frame != self.current_frame:
            self._render_frame(new_frame)

    def _on_speed_change(self, event):
        """User picked a new speed from the dropdown."""
        speed_str = self.speed_var.get()
        # e.g., "0.25x" -> 0.25
        self.speed = float(speed_str.rstrip("x"))
        # If currently playing, restart the timer with the new interval.
        # The next _schedule_next_frame call will use the new self.speed.

    def _on_channel_change(self, event):
        """User picked a new channel."""
        new_channel = self.channel_var.get()
        new_idx = self.channel_names.index(new_channel)
        if new_idx == self.channel_idx:
            return
        self.channel_idx = new_idx
        # Channel change does NOT require re-extraction (same feature matrix),
        # but DOES require rebuilding the artists since trajectory and
        # scatter plots show channel-specific columns.
        was_playing = self.is_playing
        self._pause()
        self._init_artists_for_current_path()
        self._render_frame(self.current_frame)
        if was_playing:
            self._play()

    def _on_path_change(self, event):
        """User picked a new feature path."""
        new_path = self.path_var.get()
        if new_path == self.path:
            return

        # Try to load. If Path B is disabled, this returns False.
        if not self._load_or_extract(new_path):
            # Revert dropdown to current path.
            self.path_var.set(self.path)
            return

        self.path = new_path
        # Rebuild artists (scatter pairings differ between paths).
        was_playing = self.is_playing
        self._pause()
        self._init_artists_for_current_path()
        self._render_frame(0)
        if was_playing:
            self._play()

    def _on_close(self):
        """Window close: cancel any pending after() callbacks before destroying."""
        if self._after_id is not None:
            self.window.after_cancel(self._after_id)
            self._after_id = None
        self.window.destroy()
