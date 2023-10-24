#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=nanoGPT-colab
export FLATPACK_NAME=nanogpt-scratch
# === END USER CUSTOMIZATION ===

source "$SCRIPT_DIR/device.sh" || {
  echo "⚠️ Error: Failed to source device.sh" >&2
  exit 1
}

# === BEGIN USER CUSTOMIZATION ===
# === END USER CUSTOMIZATION ===
