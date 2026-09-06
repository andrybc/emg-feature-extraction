"""
compare_features_app.py
=======================
Interactive comparison of EMG features: a clean recording versus the
same recording with the sweat artifact applied.

Feature sets:
  Path A (Hudgins) - 4 features/channel: MAV, WL, ZC, SSC. The classic
                     literature baseline.
  CAPSAS (5-feat)  - 5 features/channel: MAV, population variance, WL,
                     ZC, SSC. The lab's CAPSAS window-sweep set (40 total
                     for 8 channels), matching the Meta brainstorm slides.

Two view modes:
  Trajectory - For one channel, plot each feature over windows, clean vs
               sweaty. Shows HOW each feature changes under the artifact.
  Clusters   - Project every window's full feature vector to 2D with PCA,
               scatter colored by gesture, clean vs sweaty, with a
               separability score for each side. Shows WHETHER the
               gestures stay separable under the artifact.

Controls: Feature-set dropdown, View dropdown, Channel dropdown
(Trajectory only), Sweat slider, and a ZC/SSC threshold field. Graphs
recompute live.

Threshold and units:
  ZC and SSC use a noise threshold so tiny wiggles are not counted. The
  right value depends on the signal's amplitude scale. This simulator's
  output is in arbitrary normalized units (gestures peak near 1.0), so a
  threshold near 0.01 works well. For REAL recordings in true microvolts,
  set the threshold to the lab's 2 uV convention in the data's units. The
  field is editable so the same tool serves both.

Because the artifact is applied to the SAME loaded signal, any
difference between the two lines is purely the artifact's effect.

Run:
    PYTHONPATH=. python compare_features_app.py
"""

import os

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from src.features.data_loading import load_emg_csv, assign_window_labels
from src.features.pipeline import extract_features_path_a, extract_features_capsas
from src.artifacts.sweat import SweatProcessor


WINDOW_MS = 200
STEP_MS = 50
DEFAULT_THRESHOLD = 0.01

# Feature-set definitions: label -> (extractor fn, per-channel feature labels)
FEATURE_SETS = {
    "Path A (Hudgins)": (extract_features_path_a, ["MAV", "WL", "ZC", "SSC"]),
    "CAPSAS (5-feat)":  (extract_features_capsas, ["MAV", "VAR", "WL", "ZC", "SSC"]),
}

BG = "#0f172a"
FG = "#e2e8f0"
ACCENT = "#38bdf8"

GESTURE_COLORS = [
    "#E69F00", "#56B4E9", "#009E73",
    "#D55E00", "#CC79A7", "#0072B2",
    "#F0E442", "#000000",
]


class CompareFeaturesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Feature Comparison: clean vs sweat")
        self.root.geometry("1180x900")
        self.root.configure(bg=BG)

        self.clean = None
        self.gestures = None
        self.fs = None
        self.n_channels = 0
        self.channel_idx = 0
        self.sweat_level = 0.5
        self.threshold = DEFAULT_THRESHOLD
        self.mode = "Trajectory"
        self.feature_set = "CAPSAS (5-feat)"   # default to the slide's set

        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=12, pady=(10, 4))

        tk.Button(
            top, text="Load clean recording...", command=self._browse,
            bg="#1e293b", fg=ACCENT, activebackground="#263d5e",
            relief="flat", font=("Helvetica", 10, "bold"),
            cursor="hand2", padx=10, pady=6,
        ).pack(side="left")

        self.file_label = tk.Label(
            top, text="No file loaded", bg=BG, fg="#475569",
            font=("Helvetica", 9),
        )
        self.file_label.pack(side="left", padx=12)

        row2 = tk.Frame(self.root, bg=BG)
        row2.pack(fill="x", padx=12, pady=(0, 4))

        tk.Label(row2, text="Features:", bg=BG, fg=FG,
                 font=("Helvetica", 9)).pack(side="left", padx=(0, 4))
        self.fset_var = tk.StringVar(value=self.feature_set)
        fset_menu = ttk.Combobox(
            row2, textvariable=self.fset_var, state="readonly", width=16,
            values=list(FEATURE_SETS.keys()),
        )
        fset_menu.pack(side="left")
        fset_menu.bind("<<ComboboxSelected>>", self._on_fset_change)

        tk.Label(row2, text="View:", bg=BG, fg=FG,
                 font=("Helvetica", 9)).pack(side="left", padx=(16, 4))
        self.mode_var = tk.StringVar(value=self.mode)
        mode_menu = ttk.Combobox(
            row2, textvariable=self.mode_var, state="readonly", width=11,
            values=["Trajectory", "Clusters"],
        )
        mode_menu.pack(side="left")
        mode_menu.bind("<<ComboboxSelected>>", self._on_mode_change)

        tk.Label(row2, text="Channel:", bg=BG, fg=FG,
                 font=("Helvetica", 9)).pack(side="left", padx=(16, 4))
        self.channel_var = tk.StringVar(value="1")
        self.channel_menu = ttk.Combobox(
            row2, textvariable=self.channel_var, state="readonly",
            width=5, values=[],
        )
        self.channel_menu.pack(side="left")
        self.channel_menu.bind("<<ComboboxSelected>>", self._on_channel_change)

        tk.Label(row2, text="Sweat:", bg=BG, fg=FG,
                 font=("Helvetica", 9)).pack(side="left", padx=(16, 4))
        self.sweat_scale = tk.Scale(
            row2, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
            length=150, bg=BG, fg=FG, troughcolor="#1e293b",
            highlightthickness=0, showvalue=False,
            command=self._on_sweat_change,
        )
        self.sweat_scale.set(self.sweat_level)
        self.sweat_scale.pack(side="left")
        self.sweat_label = tk.Label(
            row2, text=f"{int(self.sweat_level*100)}%", bg=BG, fg=ACCENT,
            font=("Courier", 10, "bold"), width=5,
        )
        self.sweat_label.pack(side="left", padx=(6, 0))

        tk.Label(row2, text="thr:", bg=BG, fg=FG,
                 font=("Helvetica", 9)).pack(side="left", padx=(16, 4))
        self.thr_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
        thr_entry = tk.Entry(row2, textvariable=self.thr_var, width=8,
                             bg="#1e293b", fg=FG, insertbackground=FG,
                             relief="flat")
        thr_entry.pack(side="left")
        thr_entry.bind("<Return>", self._on_threshold_change)
        tk.Button(
            row2, text="apply", command=self._on_threshold_change,
            bg="#1e293b", fg=ACCENT, activebackground="#263d5e",
            relief="flat", font=("Helvetica", 8), cursor="hand2",
            padx=6, pady=2,
        ).pack(side="left", padx=(4, 0))

        self.fig = Figure(figsize=(11.4, 7.2), dpi=100, facecolor="white",
                          layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=12, pady=4)

        self.status = tk.Label(
            self.root, text="Load a clean recording to begin.",
            bg="#0a1628", fg="#475569", anchor="w",
            font=("Courier", 9), padx=10, pady=4,
        )
        self.status.pack(fill="x", side="bottom")

        self._show_empty_message()

    def _show_empty_message(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "Load a recording", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#999999")
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()

    def _n_per_channel(self):
        return len(FEATURE_SETS[self.feature_set][1])

    def _labels_per_channel(self):
        return FEATURE_SETS[self.feature_set][1]

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select a clean EMG recording (sweat off)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = load_emg_csv(path)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load CSV:\n{e}")
            return

        self.clean = data["signal"]
        self.gestures = data["gestures"]
        self.n_channels = data["n_channels"]
        self.fs = data["fs_estimate"] or 1000.0

        label = (f"{os.path.basename(path)}  ({self.clean.shape[0]} samples, "
                 f"{self.n_channels} ch, {self.fs:.0f} Hz")
        label += ", labeled)" if self.gestures is not None else ", no labels)"
        self.file_label.config(text=label, fg="#34d399")

        self.channel_menu.config(
            values=[str(c + 1) for c in range(self.n_channels)])
        self.channel_var.set("1")
        self.channel_idx = 0

        self._recompute_and_plot()

    def _on_fset_change(self, _e=None):
        self.feature_set = self.fset_var.get()
        self._recompute_and_plot()

    def _on_mode_change(self, _e=None):
        self.mode = self.mode_var.get()
        state = "readonly" if self.mode == "Trajectory" else "disabled"
        self.channel_menu.config(state=state)
        self._recompute_and_plot()

    def _on_channel_change(self, _e=None):
        self.channel_idx = int(self.channel_var.get()) - 1
        if self.mode == "Trajectory":
            self._recompute_and_plot()

    def _on_sweat_change(self, value_str):
        self.sweat_level = float(value_str)
        self.sweat_label.config(text=f"{int(self.sweat_level*100)}%")
        if self.clean is not None:
            self._recompute_and_plot()

    def _on_threshold_change(self, _e=None):
        try:
            self.threshold = float(self.thr_var.get())
        except ValueError:
            messagebox.showwarning("Bad threshold", "Enter a number, e.g. 0.01")
            return
        if self.clean is not None:
            self._recompute_and_plot()

    def _extract_both(self):
        window = int(round(WINDOW_MS * self.fs / 1000))
        step = int(round(STEP_MS * self.fs / 1000))
        extractor = FEATURE_SETS[self.feature_set][0]

        sp = SweatProcessor(n_channels=self.n_channels, fs=self.fs,
                            tau_sweat=0.0, seed=42)
        sp.set_sweat_level(self.sweat_level)
        sweaty = sp.process(self.clean.copy(), self.fs)

        fm_clean, _ = extractor(
            self.clean, window, step,
            zc_threshold=self.threshold, ssc_threshold=self.threshold)
        fm_sweaty, _ = extractor(
            sweaty, window, step,
            zc_threshold=self.threshold, ssc_threshold=self.threshold)

        if self.gestures is not None:
            labels = assign_window_labels(self.gestures, window, step,
                                          mode="majority")
        else:
            labels = None
        return fm_clean, fm_sweaty, labels

    def _recompute_and_plot(self):
        if self.clean is None:
            return
        if self.mode == "Trajectory":
            self._plot_trajectory()
        else:
            self._plot_clusters()

    def _plot_trajectory(self):
        fm_clean, fm_sweaty, _ = self._extract_both()
        npc = self._n_per_channel()
        flabels = self._labels_per_channel()
        base = npc * self.channel_idx

        # Choose a subplot grid that fits the feature count (4 -> 2x2, 5 -> 2x3).
        if npc <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3

        self.fig.clear()
        for i in range(npc):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            col = base + i
            ax.plot(fm_clean[:, col], label="clean",
                    color="#2563eb", linewidth=1.3)
            ax.plot(fm_sweaty[:, col],
                    label=f"sweat {int(self.sweat_level*100)}%",
                    color="#f97316", linewidth=1.3)
            ax.set_title(flabels[i], fontsize=10)
            ax.set_xlabel("window index", fontsize=8)
            ax.set_ylabel(flabels[i], fontsize=8)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
        self.fig.suptitle(
            f"{self.feature_set}  |  channel {self.channel_idx + 1}, "
            f"window {WINDOW_MS} ms, fs {self.fs:.0f} Hz, thr {self.threshold:g}",
            fontsize=11)
        self.canvas.draw_idle()

        shifts = []
        for i, lbl in enumerate(flabels):
            col = base + i
            c, s = fm_clean[:, col].mean(), fm_sweaty[:, col].mean()
            pct = 100 * (s - c) / c if c else 0.0
            shifts.append(f"{lbl} {pct:+.0f}%")
        self.status.config(
            text=f"Channel {self.channel_idx + 1}, sweat "
                 f"{int(self.sweat_level*100)}%   |   mean shift:  "
                 + "   ".join(shifts))

    def _plot_clusters(self):
        fm_clean, fm_sweaty, labels = self._extract_both()

        self.fig.clear()

        if labels is None:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5,
                    "This recording has no gesture labels.\n"
                    "The cluster view needs a gesture column.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#999999")
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw_idle()
            self.status.config(text="No gesture labels: cluster view unavailable.")
            return

        labels = np.asarray(labels)

        scaler = StandardScaler().fit(fm_clean)
        pca = PCA(n_components=2).fit(scaler.transform(fm_clean))
        proj_clean = pca.transform(scaler.transform(fm_clean))
        proj_sweaty = pca.transform(scaler.transform(fm_sweaty))

        classes = sorted(set(labels))
        color_of = {g: GESTURE_COLORS[i % len(GESTURE_COLORS)]
                    for i, g in enumerate(classes)}
        pt_colors = np.array([color_of[g] for g in labels])

        allx = np.concatenate([proj_clean[:, 0], proj_sweaty[:, 0]])
        ally = np.concatenate([proj_clean[:, 1], proj_sweaty[:, 1]])
        xpad = (allx.max() - allx.min()) * 0.05 + 1e-9
        ypad = (ally.max() - ally.min()) * 0.05 + 1e-9
        xlim = (allx.min() - xpad, allx.max() + xpad)
        ylim = (ally.min() - ypad, ally.max() + ypad)

        sep_clean = self._separability(fm_clean, labels)
        sep_sweaty = self._separability(fm_sweaty, labels)

        ax1 = self.fig.add_subplot(1, 2, 1)
        ax1.scatter(proj_clean[:, 0], proj_clean[:, 1], c=pt_colors,
                    s=18, alpha=0.7, edgecolors="none")
        ax1.set_title(f"Clean   (separability {sep_clean:.0%})", fontsize=11)
        ax1.set_xlabel("PC1", fontsize=9); ax1.set_ylabel("PC2", fontsize=9)
        ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.grid(alpha=0.3)

        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.scatter(proj_sweaty[:, 0], proj_sweaty[:, 1], c=pt_colors,
                    s=18, alpha=0.7, edgecolors="none")
        ax2.set_title(f"Sweat {int(self.sweat_level*100)}%   "
                      f"(separability {sep_sweaty:.0%})", fontsize=11)
        ax2.set_xlabel("PC1", fontsize=9); ax2.set_ylabel("PC2", fontsize=9)
        ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.grid(alpha=0.3)

        handles = [Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=color_of[g], markeredgecolor="none",
                          markersize=7, label=str(g)) for g in classes]
        ax2.legend(handles=handles, fontsize=8, loc="best", title="gesture")

        self.fig.suptitle(
            f"{self.feature_set}  |  gesture clusters in PCA space "
            f"(fit on clean), fs {self.fs:.0f} Hz, thr {self.threshold:g}",
            fontsize=11)
        self.canvas.draw_idle()

        drop = sep_clean - sep_sweaty
        self.status.config(
            text=f"{self.feature_set}   |   separability (5-fold k-NN):  "
                 f"clean {sep_clean:.0%}  ->  sweat "
                 f"{int(self.sweat_level*100)}% : {sep_sweaty:.0%}   "
                 f"(drop {drop:+.0%})")

    def _separability(self, features, labels):
        """5-fold cross-validated k-NN accuracy on standardized features."""
        n = len(labels)
        if n < 10 or len(set(labels)) < 2:
            return float("nan")
        X = StandardScaler().fit_transform(features)
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X, labels, cv=5)
        return float(scores.mean())


def main():
    root = tk.Tk()
    CompareFeaturesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()