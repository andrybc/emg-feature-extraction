"""
feature_extractor_app.py
========================
Desktop GUI for converting EMG simulator CSV files into feature matrices.

This app is the second stage of the pipeline. The first stage is the
EMG simulator (writes raw signal CSVs). This stage reads those CSVs
and produces a "feature matrix" CSV that's ready for a classifier.

Workflow:
  1. Browse... and select a CSV from the simulator.
  2. Review the auto-detected channels, sample rate, and gestures.
  3. Adjust window size / step size if needed (defaults are good).
  4. Pick a feature path:
       - Path A (Hudgins): always available
       - Path B (Phinyomark): requires enabling in src/pipeline.py first
       - Both: requires Path B enabled
  5. Click Extract Features.
  6. Preview the result, then Save to a new CSV.

Layout uses tkinter + ttk to match the simulator app's style.
"""

import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Make sure we can import the src package when running the app from any
# directory. We add the script's own directory to sys.path.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.data_loading import load_emg_csv, assign_window_labels
from src.pipeline import (
    extract_features_path_a,
    extract_features_path_b,
    extract_features_both,
)
from src.visualizer import FeatureVisualizerWindow


# ---------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------

class FeatureExtractorApp:
    """Top-level tkinter app for EMG feature extraction."""

    def __init__(self, root):
        self.root = root
        self.root.title("EMG Feature Extractor")
        self.root.geometry("900x780")

        # ----- App state -----
        # Holds the loaded data dict from data_loading.load_emg_csv,
        # or None if nothing's loaded.
        self.loaded_data = None
        # Holds the most recent feature matrix + names tuple,
        # or None if extraction hasn't been run yet.
        self.last_features = None
        self.last_feature_names = None
        self.last_window_labels = None

        # ----- Build the UI -----
        self._build_ui()

    def _build_ui(self):
        """Construct all widgets and lay them out."""
        # Main container with padding so things aren't crammed against edges.
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ===== Section 1: File selection =====
        file_frame = ttk.LabelFrame(main, text="1. Input File", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        # The selected path is shown in a read-only entry next to a Browse button.
        self.file_path_var = tk.StringVar(value="(no file selected)")
        ttk.Label(file_frame, text="CSV file:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_path_var,
                  state="readonly", width=70).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse...",
                   command=self._browse_file).grid(row=0, column=2)

        # ===== Section 2: Data info (filled in after load) =====
        info_frame = ttk.LabelFrame(main, text="2. Detected Data Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        # We use a single multi-line label that gets rewritten when a file loads.
        self.info_text_var = tk.StringVar(value="Load a file to see info.")
        ttk.Label(info_frame, textvariable=self.info_text_var,
                  justify=tk.LEFT).pack(anchor=tk.W)

        # ===== Section 3: Windowing parameters =====
        param_frame = ttk.LabelFrame(main, text="3. Windowing Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        # Window size in milliseconds. The literature standard is 150-250 ms.
        # We default to 200 ms, which is what Hudgins used.
        ttk.Label(param_frame, text="Window size (ms):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.window_ms_var = tk.IntVar(value=200)
        ttk.Spinbox(param_frame, from_=50, to=1000, increment=10,
                    textvariable=self.window_ms_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        # Step size in ms. Smaller step = more overlap = more training examples
        # but more redundancy.
        ttk.Label(param_frame, text="Step size (ms):").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.step_ms_var = tk.IntVar(value=50)
        ttk.Spinbox(param_frame, from_=10, to=500, increment=10,
                    textvariable=self.step_ms_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=(0, 20))

        # Sample rate. Auto-filled from the CSV if a timestamp column is
        # present; otherwise the user must specify it (1000 Hz default).
        ttk.Label(param_frame, text="Sample rate (Hz):").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.fs_var = tk.IntVar(value=1000)
        ttk.Spinbox(param_frame, from_=100, to=10000, increment=100,
                    textvariable=self.fs_var, width=8).grid(row=0, column=5, sticky=tk.W)

        # Label-assignment mode (only matters if the CSV has gesture labels).
        ttk.Label(param_frame, text="Window label:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        self.label_mode_var = tk.StringVar(value="majority")
        ttk.Radiobutton(param_frame, text="Majority", variable=self.label_mode_var,
                        value="majority").grid(row=1, column=1, sticky=tk.W, pady=(8, 0))
        ttk.Radiobutton(param_frame, text="Last sample", variable=self.label_mode_var,
                        value="last").grid(row=1, column=2, sticky=tk.W, pady=(8, 0))

        # ===== Section 4: Feature path selection =====
        path_frame = ttk.LabelFrame(main, text="4. Feature Set", padding=10)
        path_frame.pack(fill=tk.X, pady=(0, 10))

        # Radio buttons because the three options are mutually exclusive.
        self.feature_path_var = tk.StringVar(value="A")
        ttk.Radiobutton(
            path_frame,
            text="Path A: Hudgins (MAV, WL, ZC, SSC) - 4 features per channel",
            variable=self.feature_path_var, value="A"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            path_frame,
            text="Path B: Phinyomark (MAV, WL, WAMP, AR-4) - 7 features per channel  [requires enabling in src/pipeline.py]",
            variable=self.feature_path_var, value="B"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            path_frame,
            text="Both paths concatenated - 11 features per channel  [requires Path B enabled]",
            variable=self.feature_path_var, value="BOTH"
        ).pack(anchor=tk.W)

        # ===== Section 5: Thresholds (advanced) =====
        thr_frame = ttk.LabelFrame(main, text="5. Noise Thresholds (advanced)", padding=10)
        thr_frame.pack(fill=tk.X, pady=(0, 10))

        # These are floats in scientific notation, so we use Entry not Spinbox.
        ttk.Label(thr_frame, text="ZC threshold:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.zc_thr_var = tk.StringVar(value="1e-5")
        ttk.Entry(thr_frame, textvariable=self.zc_thr_var, width=10).grid(row=0, column=1, padx=(0, 20))

        ttk.Label(thr_frame, text="SSC threshold:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.ssc_thr_var = tk.StringVar(value="1e-5")
        ttk.Entry(thr_frame, textvariable=self.ssc_thr_var, width=10).grid(row=0, column=3, padx=(0, 20))

        ttk.Label(thr_frame, text="WAMP threshold:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.wamp_thr_var = tk.StringVar(value="1e-5")
        ttk.Entry(thr_frame, textvariable=self.wamp_thr_var, width=10).grid(row=0, column=5)

        # ===== Section 6: Action buttons =====
        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, pady=(0, 10))

        self.extract_button = ttk.Button(
            action_frame, text="Extract Features",
            command=self._on_extract_clicked, state=tk.DISABLED
        )
        self.extract_button.pack(side=tk.LEFT, padx=(0, 10))

        # Visualize button: opens the animated visualizer window. Disabled
        # until extraction succeeds (the visualizer needs windowing params
        # that match what was just extracted).
        self.visualize_button = ttk.Button(
            action_frame, text="Visualize",
            command=self._on_visualize_clicked, state=tk.DISABLED
        )
        self.visualize_button.pack(side=tk.LEFT, padx=(0, 10))

        self.save_button = ttk.Button(
            action_frame, text="Save Feature Matrix...",
            command=self._on_save_clicked, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT)

        # ===== Section 7: Status bar =====
        self.status_var = tk.StringVar(value="Ready.")
        status_label = ttk.Label(main, textvariable=self.status_var,
                                 relief=tk.SUNKEN, anchor=tk.W, padding=4)
        status_label.pack(fill=tk.X, pady=(0, 10))

        # ===== Section 8: Preview =====
        preview_frame = ttk.LabelFrame(main, text="6. Feature Matrix Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        # A ttk.Treeview gives us a quick spreadsheet-style preview.
        # Columns will be configured dynamically once features exist.
        self.preview = ttk.Treeview(preview_frame, show="headings", height=8)
        self.preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Horizontal scrollbar because feature matrices are wide.
        h_scroll = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL,
                                  command=self.preview.xview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview.configure(xscrollcommand=h_scroll.set)

    # -----------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------

    def _browse_file(self):
        """Open file dialog and load the selected CSV."""
        path = filedialog.askopenfilename(
            title="Select EMG CSV from simulator",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return  # User cancelled.

        try:
            self.status_var.set(f"Loading {os.path.basename(path)}...")
            self.root.update_idletasks()

            data = load_emg_csv(path)
            self.loaded_data = data
            self.file_path_var.set(path)

            # Build the info text block.
            info_lines = [
                f"Channels detected: {data['n_channels']}  -->  {', '.join(data['channels'])}",
                f"Samples: {data['n_samples']}",
            ]
            if data["fs_estimate"]:
                info_lines.append(
                    f"Sample rate (estimated from timestamps): {data['fs_estimate']:.1f} Hz"
                )
                # Auto-fill the sample rate field if we estimated it.
                self.fs_var.set(int(round(data["fs_estimate"])))
            else:
                info_lines.append(
                    "Sample rate: no timestamp column found, using value below"
                )
            duration_sec = data["n_samples"] / self.fs_var.get()
            info_lines.append(f"Approx duration: {duration_sec:.2f} sec")

            if data["gestures"] is not None:
                unique_gestures = pd.unique(data["gestures"])
                info_lines.append(
                    f"Gesture labels found: {', '.join(str(g) for g in unique_gestures)}"
                )
            else:
                info_lines.append(
                    "Gesture labels: none found (output will have no label column)"
                )

            self.info_text_var.set("\n".join(info_lines))
            self.status_var.set("File loaded successfully.")
            self.extract_button.configure(state=tk.NORMAL)
            # Clear any previous extraction results.
            self.save_button.configure(state=tk.DISABLED)
            self.visualize_button.configure(state=tk.DISABLED)
            self._clear_preview()

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load CSV:\n{e}")
            self.status_var.set("Load failed.")
            self.loaded_data = None
            self.extract_button.configure(state=tk.DISABLED)

    def _on_extract_clicked(self):
        """Run feature extraction in a background thread so the UI stays responsive."""
        if self.loaded_data is None:
            messagebox.showwarning("No data", "Load a CSV file first.")
            return

        # Disable the button so users can't click it twice.
        self.extract_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.status_var.set("Extracting features...")

        # Run extraction on a worker thread to keep the GUI responsive.
        # tkinter is not thread-safe for widget updates, so we use root.after
        # to marshal completion back to the main thread.
        thread = threading.Thread(target=self._do_extraction, daemon=True)
        thread.start()

    def _do_extraction(self):
        """Worker function (runs off the main thread)."""
        try:
            # Pull config values from the UI vars.
            fs = self.fs_var.get()
            window_ms = self.window_ms_var.get()
            step_ms = self.step_ms_var.get()

            # Convert milliseconds to samples. This is the most common
            # source of off-by-one errors in EMG pipelines, so be careful:
            #   samples = ms * fs / 1000
            window_size = int(round(window_ms * fs / 1000))
            step_size = int(round(step_ms * fs / 1000))

            zc_thr = float(self.zc_thr_var.get())
            ssc_thr = float(self.ssc_thr_var.get())
            wamp_thr = float(self.wamp_thr_var.get())

            signal = self.loaded_data["signal"]
            gestures = self.loaded_data["gestures"]
            channels = self.loaded_data["channels"]
            path = self.feature_path_var.get()

            # Dispatch to the chosen pipeline function. We pass channel
            # names through so the output column names trace back to the
            # original input columns (e.g. 'ch1_mV_mav' not 'ch0_mav').
            if path == "A":
                fm, names = extract_features_path_a(
                    signal, window_size, step_size,
                    zc_threshold=zc_thr, ssc_threshold=ssc_thr,
                    channel_names=channels,
                )
            elif path == "B":
                fm, names = extract_features_path_b(
                    signal, window_size, step_size,
                    wamp_threshold=wamp_thr,
                    channel_names=channels,
                )
            else:  # BOTH
                fm, names = extract_features_both(
                    signal, window_size, step_size,
                    zc_threshold=zc_thr,
                    ssc_threshold=ssc_thr,
                    wamp_threshold=wamp_thr,
                    channel_names=channels,
                )

            # Assign window labels if we have per-sample gestures.
            if gestures is not None:
                window_labels = assign_window_labels(
                    gestures, window_size, step_size,
                    mode=self.label_mode_var.get()
                )
            else:
                window_labels = None

            # Stash results and trigger the UI update on the main thread.
            self.last_features = fm
            self.last_feature_names = names
            self.last_window_labels = window_labels

            self.root.after(0, self._on_extraction_done, None)

        except Exception as e:
            # Capture the traceback now (we're still on the worker thread).
            tb = traceback.format_exc()
            self.root.after(0, self._on_extraction_done, (e, tb))

    def _on_extraction_done(self, error_info):
        """Runs on the main thread after extraction finishes."""
        if error_info is not None:
            e, tb = error_info
            messagebox.showerror(
                "Extraction Error",
                f"Feature extraction failed:\n\n{e}\n\nDetails:\n{tb}"
            )
            self.status_var.set("Extraction failed.")
            self.extract_button.configure(state=tk.NORMAL)
            return

        fm = self.last_features
        names = self.last_feature_names
        self.status_var.set(
            f"Done. Feature matrix shape: {fm.shape[0]} windows x {fm.shape[1]} features."
        )
        self._populate_preview(fm, names, self.last_window_labels)
        self.extract_button.configure(state=tk.NORMAL)
        self.save_button.configure(state=tk.NORMAL)
        # Enable visualizer now that we have a valid extraction. The
        # visualizer will re-extract internally if the user changes path,
        # but we need at least one successful run to confirm the windowing
        # params are sane.
        self.visualize_button.configure(state=tk.NORMAL)

    def _on_visualize_clicked(self):
        """Open the animated feature visualizer window."""
        if self.loaded_data is None or self.last_features is None:
            messagebox.showwarning(
                "Not ready",
                "Load a file and run Extract Features first."
            )
            return

        # Compute the same windowing params we just used for extraction.
        fs = self.fs_var.get()
        window_ms = self.window_ms_var.get()
        step_ms = self.step_ms_var.get()
        window_size = int(round(window_ms * fs / 1000))
        step_size = int(round(step_ms * fs / 1000))

        # Parse thresholds (re-using the same fields as Extract).
        try:
            zc_thr = float(self.zc_thr_var.get())
            ssc_thr = float(self.ssc_thr_var.get())
            wamp_thr = float(self.wamp_thr_var.get())
        except ValueError:
            zc_thr = ssc_thr = wamp_thr = 1e-5

        # Pass the user's current path selection so the visualizer starts
        # in the same mode they extracted with.
        initial_path = self.feature_path_var.get()

        # Construct and show. The visualizer manages its own lifecycle.
        FeatureVisualizerWindow(
            parent=self.root,
            loaded_data=self.loaded_data,
            window_size_samples=window_size,
            step_size_samples=step_size,
            fs=fs,
            initial_path=initial_path,
            zc_threshold=zc_thr,
            ssc_threshold=ssc_thr,
            wamp_threshold=wamp_thr,
            label_mode=self.label_mode_var.get(),
        )

    def _on_save_clicked(self):
        """Save the most recent feature matrix to a CSV."""
        if self.last_features is None:
            return

        # Suggest a default filename based on the input file.
        default_name = "features.csv"
        if self.loaded_data:
            input_path = self.file_path_var.get()
            stem = Path(input_path).stem
            path_tag = self.feature_path_var.get().lower()
            default_name = f"{stem}_features_path{path_tag}.csv"

        out_path = filedialog.asksaveasfilename(
            title="Save feature matrix as",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")]
        )
        if not out_path:
            return  # User cancelled.

        try:
            # Build a DataFrame so pandas handles the header row and escaping.
            df = pd.DataFrame(self.last_features, columns=self.last_feature_names)
            # Add the label column if we have one. We put it LAST so the
            # feature columns stay in the same positions whether or not
            # labels are present (helpful for downstream code).
            if self.last_window_labels is not None:
                df["gesture"] = self.last_window_labels

            df.to_csv(out_path, index=False)
            self.status_var.set(f"Saved to {out_path}")
            messagebox.showinfo("Saved", f"Feature matrix saved to:\n{out_path}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save:\n{e}")

    # -----------------------------------------------------------------
    # Preview helpers
    # -----------------------------------------------------------------

    def _clear_preview(self):
        """Remove all rows and columns from the preview table."""
        for child in self.preview.get_children():
            self.preview.delete(child)
        self.preview["columns"] = ()

    def _populate_preview(self, feature_matrix, feature_names, window_labels):
        """Fill the Treeview with the first ~20 rows of the feature matrix."""
        self._clear_preview()

        # Configure columns. We add 'window' as an index column at the start
        # and 'gesture' at the end if labels are available.
        columns = ["window"] + list(feature_names)
        if window_labels is not None:
            columns.append("gesture")

        self.preview["columns"] = columns
        for col in columns:
            self.preview.heading(col, text=col)
            # Use a modest column width. Numeric columns are usually short.
            self.preview.column(col, width=80, minwidth=50, anchor=tk.E)

        # Show the first 20 rows (no point dumping thousands into a GUI).
        n_preview = min(20, feature_matrix.shape[0])
        for i in range(n_preview):
            # Format floats to 4 decimal places for readability.
            row_vals = [i] + [f"{v:.4f}" for v in feature_matrix[i]]
            if window_labels is not None:
                row_vals.append(str(window_labels[i]))
            self.preview.insert("", tk.END, values=row_vals)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = FeatureExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
