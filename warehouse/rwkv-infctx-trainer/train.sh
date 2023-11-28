#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=RWKV-infctx-trainer
export FLATPACK_NAME=rwkv-infctx-trainer
# === END USER CUSTOMIZATION ===

source "$SCRIPT_DIR/device.sh" || {
  echo "⚠️ Error: Failed to source device.sh" >&2
  exit 1
}

# === BEGIN USER CUSTOMIZATION ===
mkdir -p checkpoint/
mkdir -p datapath/
mkdir -p dataset/dataset-config/text/
mkdir -p dataset/dataset-config/zip/
mkdir -p model/

cd RWKV-v4neo || exit
"${VENV_PYTHON}" ./init_model.py --n_layer 6 --n_embd 512 --vocab_size neox --skip-if-exists ../model/L6-D512-neox-init.pth
cd /content/rwkv-infctx-trainer/RWKV-infctx-trainer/RWKV-v4neo || exit
"${VENV_PYTHON}" ./init_model.py --n_layer 6 --n_embd 512 --vocab_size world --skip-if-exists ../model/L6-D512-world-init.pth
# === END USER CUSTOMIZATION ===
