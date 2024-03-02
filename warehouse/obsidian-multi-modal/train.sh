#!/bin/bash

# Get the directory where the script is located
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=llama.cpp
export FLATPACK_NAME=obsidian-multi-modal
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
make

mkdir -p models

if [ ! -f models/mmproj-obsidian-f16.gguf ]; then
    wget -O models/mmproj-obsidian-f16.gguf https://huggingface.co/NousResearch/Obsidian-3B-V0.5-GGUF/resolve/main/mmproj-obsidian-f16.gguf
else
    echo "File 'models/mmproj-obsidian-f16.gguf' already exists, skipping download."
fi

if [ ! -f models/obsidian-q6.gguf ]; then
    wget -O models/obsidian-q6.gguf https://huggingface.co/NousResearch/Obsidian-3B-V0.5-GGUF/resolve/main/obsidian-q6.gguf
else
    echo "File 'models/obsidian-q6.gguf' already exists, skipping download."
fi
# === END USER CUSTOMIZATION ===
