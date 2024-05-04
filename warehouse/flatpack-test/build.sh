#!/bin/bash

# Get the directory where the script is located
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 build.sh is running in: $SCRIPT_DIR"

# === BEGIN USER CUSTOMIZATION ===
export DEFAULT_REPO_NAME=llama.cpp
export FLATPACK_NAME=flatpack-test
# === END USER CUSTOMIZATION ===

source "$SCRIPT_DIR/device.sh" || {
  echo "😱 Error: Failed to source device.sh" >&2
  exit 1
}

# Required devices (cpu cuda mps)
REQUIRED_DEVICES="cuda mps"

# Check if DEVICE is among the required devices
if [[ ! " $REQUIRED_DEVICES " =~ " $DEVICE " ]]; then
  echo "😱 Error: This script requires one of the following devices: $REQUIRED_DEVICES." >&2
  exit 1
fi

# === BEGIN USER CUSTOMIZATION ===
# Check if the build was already done by looking for a build marker file
if [ -f ".build_successful" ]; then
    echo "✅ Build already completed for llama.cpp"
else
    echo "🦙 Building llama.cpp"
    # Run make and check if it succeeded
    if make; then
        echo "✅ Finished building llama.cpp"
        # Create a marker file to indicate the build was successful
        touch .build_successful
    else
        echo "❌ Build failed for llama.cpp"
        exit 1
    fi
fi
# === END USER CUSTOMIZATION ===