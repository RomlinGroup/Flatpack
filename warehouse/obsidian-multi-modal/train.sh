#!/bin/bash

# Get the directory where the script is located
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=Obsidian
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
"${VENV_PIP}" install --upgrade pip
"${VENV_PIP}" install -e .

if [[ "$OS" = "Darwin" ]]; then
  # https://developer.apple.com/metal/pytorch/
  "${VENV_PIP}" install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
fi

"${VENV_PIP}" uninstall bitsandbytes -y

"${VENV_PYTHON}" -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
# === END USER CUSTOMIZATION ===
