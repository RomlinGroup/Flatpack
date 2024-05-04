#!/bin/bash

# Check essential environment variables: DEFAULT_REPO_NAME, FLATPACK_NAME, and SCRIPT_DIR
for VAR_NAME in DEFAULT_REPO_NAME FLATPACK_NAME SCRIPT_DIR; do
  if [[ -z "${!VAR_NAME}" ]]; then
    echo "Error: $VAR_NAME is not set. Please set the $VAR_NAME environment variable." >&2
    exit 1
  fi
done

# Default user directory path
DEFAULT_PATH="/home/$(whoami)/flatpacks"
OS=$(uname)

# Environment detection and configuration
if [[ -d "/content" ]]; then
  echo "🌀 Detected Google Colab environment"
  if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
  export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
  DEFAULT_PATH="/content"

elif [ "$OS" = "Darwin" ]; then
  echo "🍎 Detected macOS environment"
  DEVICE="mps"
  export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
  if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: Python executable not found at $VENV_PYTHON" >&2
    exit 1
  fi

elif [ "$OS" = "Linux" ]; then
  echo "🐧 Detected Linux environment"
  if command -v python3 &> /dev/null; then
    export VENV_PYTHON="python3"
  elif command -v python &> /dev/null; then
    export VENV_PYTHON="python"
  else
    echo "Error: Python is not installed on this system." >&2
    exit 1
  fi
  DEVICE="cpu"
else
  echo "❓ Detected other OS environment"
  DEVICE="cpu"
fi

WORK_DIR="${DEFAULT_PATH}/${FLATPACK_NAME}/build/${DEFAULT_REPO_NAME}"
export VENV_PIP="$(dirname "$VENV_PYTHON")/pip"
if [[ ! -x "$VENV_PIP" ]]; then
  echo "Error: pip executable not found at $VENV_PIP" >&2
  exit 1
fi

# Echo the determined working directory and device
echo "Determined WORK_DIR: $WORK_DIR"
echo "Determined DEVICE: $DEVICE"

# Change directory to WORK_DIR
if [[ -d "$WORK_DIR" ]]; then
  cd "$WORK_DIR"
  echo "Current working directory after changing: $(pwd)"
  echo "[DEBUG] VENV_PYTHON: $VENV_PYTHON / VENV_PIP: $VENV_PIP"
else
  echo "Error: Failed to change to directory $WORK_DIR" >&2
  exit 1
fi
