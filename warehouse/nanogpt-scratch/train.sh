#!/bin/bash

# Get the directory where the script is located
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=nanoGPT
export FLATPACK_NAME=nanogpt-scratch
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
cp train.py train.py.backup
sed -i "s/device = 'cuda'/device = 'mps'/" train.py
sed -i "s/dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'/dtype = 'float16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'/" train.py
sed -i 's/compile = True/compile = False/' train.py
"${VENV_PYTHON}" data/shakespeare_char/prepare.py
"${VENV_PYTHON}" train.py config/train_shakespeare_char.py
# === END USER CUSTOMIZATION ===
