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
IS_COLAB="${COLAB_GPU:-0}"
OS=$(uname)
if [[ "${IS_COLAB}" == "1" ]]; then
  echo "🌀 Detected Colab environment"
  export VENV_PYTHON="python"
  WORK_DIR="/content/$FLATPACK_NAME/$REPO_NAME"
  DEVICE="cuda"
elif [ "$OS" = "Darwin" ]; then
  echo "🍎 Detected macOS environment"
  export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
  WORK_DIR="$REPO_NAME"
  DEVICE="mps"
elif [ "$OS" = "Linux" ]; then
  export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
  if [ -f /etc/os-release ] && grep -q "Ubuntu" /etc/os-release; then
    echo "🐧 Detected Ubuntu environment"
  else
    echo "🐧 Detected Linux environment (non-Ubuntu)"
  fi
  WORK_DIR="$DEFAULT_PATH/$FLATPACK_NAME"
  DEVICE="cpu"
else
  echo "❓ Detected other OS environment"
  # The same settings as non-Ubuntu Linux, so no changes here.
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
