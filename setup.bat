@echo off
REM Create the project venv and install dependencies.
REM Run once after cloning the repo.
REM
REM Usage:
REM   setup.bat

setlocal enabledelayedexpansion

set PYTHON_VERSION=3.12

REM Try the py launcher first (recommended on Windows), then fall back to python
where py >nul 2>nul
if %ERRORLEVEL% == 0 (
    set PYTHON_CMD=py -%PYTHON_VERSION%
) else (
    where python >nul 2>nul
    if !ERRORLEVEL! == 0 (
        set PYTHON_CMD=python
    ) else (
        echo Error: Python not found.
        echo Install Python %PYTHON_VERSION% from python.org and re-run this script.
        exit /b 1
    )
)

echo Using: %PYTHON_CMD%
%PYTHON_CMD% -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo Error: failed to create virtual environment.
    exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Error: failed to install dependencies.
    exit /b 1
)

echo.
echo Done. Activate the venv before running the simulator:
echo   .venv\Scripts\activate.bat
echo   set PYTHONPATH=.
echo   python emg_simulator.py

endlocal