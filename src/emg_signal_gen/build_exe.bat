@echo off
REM ============================================================
REM  build_exe.bat  --  Build EMG Simulator as a standalone .exe
REM  Run this from your repo root inside your .venv
REM ============================================================

echo.
echo  EMG Simulator -- Windows EXE Builder
echo  ======================================

REM Step 1: Make sure pyinstaller is installed
echo  [1/4] Checking PyInstaller...
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo  Installing PyInstaller...
    pip install pyinstaller
)

REM Step 2: Clean previous build artifacts
echo  [2/4] Cleaning old build files...
if exist build   rmdir /s /q build
if exist dist    rmdir /s /q dist
if exist EMGSimulator.spec del /q EMGSimulator.spec

REM Step 3: Build the .exe
echo  [3/4] Building .exe (this takes 30-90 seconds)...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name EMGSimulator ^
    --hidden-import matplotlib.backends.backend_tkagg ^
    --hidden-import scipy.signal ^
    --hidden-import numpy ^
    --collect-all matplotlib ^
    --hidden-import PIL._tkinter_finder^
    --collect-all PIL^
    emg_simulator.py

REM Step 4: Check result
echo  [4/4] Build complete!
if exist dist\EMGSimulator.exe (
    echo.
    echo  SUCCESS: dist\EMGSimulator.exe is ready
    echo  Copy it to any Windows machine -- no Python needed.
    echo.
    echo  The .exe will create a data\simulated\ folder
    echo  next to itself when you record EMG data.
) else (
    echo.
    echo  BUILD FAILED -- check the output above for errors.
)

pause
