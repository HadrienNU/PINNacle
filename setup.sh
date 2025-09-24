#!/bin/bash

echo "========================================"
echo "    PINNacle Setup Script (Linux/Mac)"
echo "========================================"

set -euo pipefail

# Choose a venv directory that matches the platform
VENV_DIR="venv"
if [ -d "${VENV_DIR}/Scripts" ]; then
    echo "Detected a Windows virtual environment at '${VENV_DIR}'."
    echo "Creating/using a separate Unix virtual environment at 'venv_unix'."
    VENV_DIR="venv_unix"
fi

# Create venv if needed
if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment in '${VENV_DIR}'..."
        python3 -m venv "${VENV_DIR}"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Install requirements
echo "Installing requirements..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

echo
echo "Setup complete. Virtual environment: '${VENV_DIR}'"
