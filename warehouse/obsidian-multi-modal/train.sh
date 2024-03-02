#!/bin/bash

# Get the directory where the script is located
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=llama-cpp-python
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
mkdir models
wget -O models/mmproj-obsidian-f16.gguf https://huggingface.co/NousResearch/Obsidian-3B-V0.5-GGUF/resolve/main/mmproj-obsidian-f16.gguf
wget -O models/obsidian-q6.gguf https://huggingface.co/NousResearch/Obsidian-3B-V0.5-GGUF/resolve/main/obsidian-q6.gguf

CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
"${VENV_PIP}" install llama-cpp-python

cp ../demo.py demo.py
"${VENV_PYTHON}" demo.py
# === END USER CUSTOMIZATION ===
