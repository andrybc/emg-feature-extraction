# EMG Simulator

Real-time 8-channel surface EMG simulator with a modular artifact processing pipeline. Developed at LIMB (Laboratory for Interaction of Machine and Brain) for myoelectric control research.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=. python emg_simulator.py
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

```bash
./build_exe.sh   # Linux/macOS
build_exe.bat    # Windows
```