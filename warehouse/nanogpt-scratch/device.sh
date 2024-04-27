#!/bin/bash

for VAR_NAME in REPO_NAME FLATPACK_NAME SCRIPT_DIR; do
  if [[ -z "${!VAR_NAME}" ]]; then
    echo "Error: $VAR_NAME is not set. Please set the $VAR_NAME environment variable." >&2
    exit 1
  fi
done

DEFAULT_PATH="/home/$(whoami)/flatpacks"
OS=$(uname)

if [[ -d "/content" ]]; then
  if command -v nvidia-smi &> /dev/null; then
    echo "🌀 Detected Colab GPU environment"
    DEVICE="cuda"
  else
    echo "🌀 Detected Colab CPU environment"
    DEVICE="cpu"
  fi

  export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
  DEFAULT_PATH="/content"
  WORK_DIR="$DEFAULT_PATH/$FLATPACK_NAME/build/$REPO_NAME"

elif [ "$OS" = "Darwin" ]; then
  echo "🍎 Detected macOS environment"
  export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
  DEFAULT_PATH="/Users/$(whoami)/flatpacks"
  WORK_DIR="$DEFAULT_PATH/$FLATPACK_NAME/build/$REPO_NAME"
  DEVICE="mps"

elif [ "$OS" = "Linux" ]; then
  if [[ -x "$(command -v python3)" ]]; then
    export VENV_PYTHON="python3"
  else
    export VENV_PYTHON="python"
  fi
  if [ -f /etc/os-release ] && grep -q "Ubuntu" /etc/os-release; then
    echo "🐧 Detected Ubuntu environment"
  else
    echo "🐧 Detected Linux environment (non-Ubuntu)"
  fi
  WORK_DIR="$DEFAULT_PATH/$FLATPACK_NAME/build/$REPO_NAME"
  DEVICE="cpu"

else
  echo "❓ Detected other OS environment"
  DEVICE="cpu"
  WORK_DIR="$DEFAULT_PATH/$FLATPACK_NAME/build/$REPO_NAME"
fi

if [[ "$OS" = "Darwin" ]] || [[ "$OS" = "Linux" ]] || [[ -d "/content" ]]; then
    export VENV_PIP="$(dirname $VENV_PYTHON)/pip"
else
    echo "⚠️  Virtual environment's pip could not be determined."
    exit 1
fi

echo "Determined WORK_DIR: $WORK_DIR"
echo "Determined DEVICE: $DEVICE"

if [[ -d "$WORK_DIR" ]]; then
  cd "$WORK_DIR"
  echo "Current working directory after changing: $(pwd)"
  echo "[DEBUG] VENV_PYTHON: $VENV_PYTHON / VENV_PIP: $VENV_PIP"

  if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ Error: Not running inside a virtual environment. Exiting." >&2
    exit 1
  fi

else
  echo "❌ Error: Failed to change to directory $WORK_DIR" >&2
  exit 1
fi
