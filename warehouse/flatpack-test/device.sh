#!/bin/bash

# Check essential environment variables: DEFAULT_REPO_NAME, FLATPACK_NAME, and SCRIPT_DIR
for VAR_NAME in DEFAULT_REPO_NAME FLATPACK_NAME SCRIPT_DIR; do
  if [[ -z "${!VAR_NAME}" ]]; then
    echo "Error: $VAR_NAME is not set. Please set the $VAR_NAME environment variable." >&2
    exit 1
  fi
done

# Set the working directory to the current directory
WORK_DIR="$(pwd)"

# Environment detection
OS=$(uname)

if [[ -d "/content" ]]; then
  # Detected Google Colab environment
  if command -v nvidia-smi &> /dev/null; then
    echo "🌀 Detected Colab GPU environment"
    DEVICE="cuda"
  else
    echo "🌀 Detected Colab CPU environment"
    DEVICE="cpu"
  fi
  export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
elif [ "$OS" = "Darwin" ]; then
  echo "🍎 Detected macOS environment"
  export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
  DEVICE="mps"
elif [ "$OS" = "Linux" ]; then
  # Check for Python version and adjust VENV_PYTHON accordingly
  if [[ -x "$(command -v python3)" ]]; then
    export VENV_PYTHON="python3"
  else
    export VENV_PYTHON="python"
  fi
  echo "🐧 Detected Linux environment"
  DEVICE="cpu"
else
  echo "❓ Detected other OS environment"
  DEVICE="cpu"  # Assume CPU for other environments as a fallback
fi

# Determine the location of 'pip' based on VENV_PYTHON
export VENV_PIP="$(dirname $VENV_PYTHON)/pip"

# Echo the determined working directory and device
echo "Determined WORK_DIR: $WORK_DIR"
echo "Determined DEVICE: $DEVICE"

# Check if the directory exists before changing
if [[ -d "$WORK_DIR" ]]; then
  cd "$WORK_DIR"
else
  echo "Error: Failed to change to directory $WORK_DIR" >&2
  exit 1
fi
