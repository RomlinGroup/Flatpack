#!/bin/bash

# Check essential environment variables: REPO_NAME, FLATPACK_NAME, and SCRIPT_DIR
for VAR_NAME in REPO_NAME FLATPACK_NAME SCRIPT_DIR; do
  if [[ -z "${!VAR_NAME}" ]]; then
    echo "Error: $VAR_NAME is not set. Please set the $VAR_NAME environment variable." >&2
    exit 1
  fi
done

# Default user directory path
DEFAULT_PATH="/home/$(whoami)"

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
  export VENV_PYTHON="/usr/bin/python3"
  WORK_DIR="/content/$FLATPACK_NAME/$REPO_NAME"
elif [ "$OS" = "Darwin" ]; then
  echo "🍎 Detected macOS environment"
  export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
  WORK_DIR="$REPO_NAME"
  DEVICE="mps"
elif [ "$OS" = "Linux" ]; then
  # Check for Python version and adjust VENV_PYTHON accordingly
  if [[ -x "$(command -v python3)" ]]; then
    export VENV_PYTHON="python3"
  else
    export VENV_PYTHON="python"
  fi
  if [ -f /etc/os-release ] && grep -q "Ubuntu" /etc/os-release; then
    echo "🐧 Detected Ubuntu environment"
    WORK_DIR="$DEFAULT_PATH/$FLATPACK_NAME/$REPO_NAME"
  else
    echo "🐧 Detected Linux environment (non-Ubuntu)"
    WORK_DIR="$DEFAULT_PATH/$FLATPACK_NAME/$REPO_NAME"
  fi
  DEVICE="cpu"
else
  echo "❓ Detected other OS environment"
  # Assume CPU for other environments as a fallback
  DEVICE="cpu"
  WORK_DIR="$DEFAULT_PATH/$FLATPACK_NAME/$REPO_NAME"
fi

# Echo the determined working directory and device
echo "Determined WORK_DIR: $WORK_DIR"
echo "Determined DEVICE: $DEVICE"

# Check if the directory exists before changing
if [[ -d "$WORK_DIR" ]]; then
  cd "$WORK_DIR"
  echo "Current working directory after changing: $(pwd)"
else
  echo "Error: Failed to change to directory $WORK_DIR" >&2
  exit 1
fi