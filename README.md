# EMG Intent Prediction Pipeline

A beginner-friendly research coding repo for learning EMG signal processing and feature extraction.

## Week 1 goal

Build a Python pipeline that turns raw or simulated EMG data into a feature matrix for later machine learning.

## Repo structure

```text
emg-intent-prediction-pipeline/
  README.md
  requirements.txt
  .gitignore
  data/
    raw/
    processed/
  notebooks/
    01_emg_signal_basics.ipynb
    02_filtering_rectification_rms.ipynb
    03_windowing_and_features.ipynb
  src/
    signal_generation.py
    preprocessing.py
    windowing.py
    features.py
    plotting.py
  notes/
    week1_emg_basics.md
    week1_signal_processing_math.md
```

## Setup

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

## Day 1

Open:

```text
notebooks/01_emg_signal_basics.ipynb
```

You will generate fake EMG-like data and plot it.
