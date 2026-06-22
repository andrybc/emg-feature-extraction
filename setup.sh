#!/bin/bash
# Create the project venv and install dependencies.
# Run once after cloning the repo.
#
# Usage:
#   ./setup.sh

set -e

PYTHON_VERSION="3.12"

if ! command -v python${PYTHON_VERSION} &> /dev/null; then
    echo "Error: python${PYTHON_VERSION} not found."
    echo "Install it via pyenv, your package manager, or python.org."
    exit 1
fi

python${PYTHON_VERSION} -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. Activate the venv before running the simulator:"
echo "  source .venv/bin/activate"
echo "  PYTHONPATH=. python emg_simulator.py"