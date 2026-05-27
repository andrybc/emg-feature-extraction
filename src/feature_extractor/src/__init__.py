"""
EMG feature extraction package.

Exposes the main entry points so the app can import them cleanly.
"""

from .features import (
    mav,
    waveform_length,
    zero_crossings,
    slope_sign_changes,
    variance,
    willison_amplitude,
    ar_coefficients,
)
from .windowing import window_signal
from .data_loading import load_emg_csv, assign_window_labels
from .pipeline import (
    extract_features_path_a,
    extract_features_path_b,
    extract_features_both,
)
from .visualizer import FeatureVisualizerWindow
