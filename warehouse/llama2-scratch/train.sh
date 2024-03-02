#!/bin/bash

# Get the directory where the script is located
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=llama2.c
export FLATPACK_NAME=llama2-scratch
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
sed -i 's/batch_size = 128/batch_size = 64/' train.py
sed -i 's/dtype = "bfloat16"/dtype = "float16"/' train.py
"${VENV_PYTHON}" tinystories.py download
"${VENV_PYTHON}" tinystories.py pretokenize
"${VENV_PYTHON}" train.py
# === END USER CUSTOMIZATION ===