# EMG Simulator

Real-time 8-channel surface EMG simulator with a modular artifact processing pipeline. Developed at LIMB (Laboratory for Interaction of Machine and Brain) for myoelectric control research.

## Quick start

Open Command Prompt (cmd.exe) in the repo root and run:

```
setup.bat
.venv\Scripts\activate.bat
set PYTHONPATH=.
python emg_simulator.py
```

## Setup

This project uses Python 3.12 (see `.python-version`). Required dependencies are pinned in `requirements.txt`.

### First-time setup

```
setup.bat
```

### Every session

```
.venv\Scripts\activate.bat
set PYTHONPATH=.
python emg_simulator.py
```

If you get `ModuleNotFoundError` for numpy, scipy, or matplotlib, you forgot to activate the venv. Re-run the activate line and try again.

### PowerShell users

If you prefer PowerShell over cmd.exe, the activate command is different:

```
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python emg_simulator.py
```

You may need to allow script execution once with:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Layout

```
emg_simulator.py          Real-time GUI simulator (Tkinter + matplotlib)
load_emg_recording.py     Load recorded or calibration CSVs back into the simulator
ninapro_to_calibration.py Convert NinaPro DB2 .mat files to calibration CSVs
src/artifacts/base.py     BaseArtifact abstract class (the contract for all artifacts)
```

## Artifact branches

Each artifact under development lives on its own branch.

| Branch            | Status     | Adds                              |
|-------------------|------------|-----------------------------------|
| `main`            | Stable     | Simulator + BaseArtifact only     |
| `artifact/sweat`  | Validated against literature | SweatProcessor + offline CLI + visualizer |

To add a new artifact: branch from `main`, subclass `BaseArtifact` in `src/artifacts/<name>.py`, wire it into the simulator's artifact list, and open a PR.

## Build a standalone executable

```
build_exe.bat
```