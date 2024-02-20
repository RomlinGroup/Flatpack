#!/bin/bash

# Get the directory where the script is located
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=template
export FLATPACK_NAME=template
# === END USER CUSTOMIZATION ===

source "$SCRIPT_DIR/device.sh" || {
  echo "⚠️ Error: Failed to source device.sh" >&2
  exit 1
}

# Use "cuda" for GPU requirement, "cpu" for CPU requirement
REQUIRED_DEVICE="cuda"

if [ "$DEVICE" != "$REQUIRED_DEVICE" ]; then
  echo "⚠️ Error: This script requires a $REQUIRED_DEVICE device." >&2
  exit 1
fi

# === BEGIN USER CUSTOMIZATION ===
"${VENV_PYTHON}" -c "print('Hello, World!')"
# === END USER CUSTOMIZATION ===
