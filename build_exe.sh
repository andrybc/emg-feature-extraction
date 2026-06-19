#!/usr/bin/env bash
# ============================================================
#  build_exe.sh  --  Build EMG Simulator as a standalone binary
#  Mac: produces a .app bundle in dist/
#  Linux: produces a single executable in dist/
#
#  Run from your repo root inside your activated .venv:
#    chmod +x build_exe.sh
#    ./build_exe.sh
# ============================================================

set -e   # Exit immediately if any command fails

echo ""
echo " EMG Simulator -- Mac/Linux Builder"
echo " ===================================="

# Step 1: Install PyInstaller if not present
echo " [1/4] Checking PyInstaller..."
if ! pip show pyinstaller &>/dev/null; then
    echo " Installing PyInstaller..."
    pip install pyinstaller
fi

# Step 2: Clean previous build artifacts
echo " [2/4] Cleaning old build files..."
rm -rf build dist EMGSimulator.spec

# Step 3: Build
echo " [3/4] Building executable (30-90 seconds)..."

pyinstaller \
    --onefile \
    --windowed \
    --name EMGSimulator \
    --hidden-import matplotlib.backends.backend_tkagg \
    --hidden-import scipy.signal \
    --hidden-import numpy \
    --collect-all matplotlib \
    --hidden-import PIL._tkinter_finder\
    --collect-all PIL\
    emg_simulator.py

# Step 4: Report
echo " [4/4] Build complete!"

if [ -f "dist/EMGSimulator" ]; then
    echo ""
    echo " SUCCESS (Linux): dist/EMGSimulator is ready"
    echo " Run with: ./dist/EMGSimulator"
elif [ -d "dist/EMGSimulator.app" ]; then
    echo ""
    echo " SUCCESS (Mac): dist/EMGSimulator.app is ready"
    echo " Double-click it in Finder or run: open dist/EMGSimulator.app"
else
    echo ""
    echo " BUILD FAILED -- check the output above for errors"
    exit 1
fi

echo ""
echo " The app will create a data/simulated/ folder next to"
echo " the executable when you record EMG data."
