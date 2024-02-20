#!/bin/bash

# Get the directory where the script is located
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=RWKV-infctx-trainer
export FLATPACK_NAME=rwkv-infctx-trainer
# === END USER CUSTOMIZATION ===

source "$SCRIPT_DIR/device.sh" || {
  echo "😱 Error: Failed to source device.sh" >&2
  exit 1
}

# Use "cuda" for GPU requirement, "cpu" for CPU requirement
REQUIRED_DEVICE="cuda"

if [ "$DEVICE" != "$REQUIRED_DEVICE" ]; then
  echo "😱 Error: This script requires a $REQUIRED_DEVICE device." >&2
  exit 1
fi

# === BEGIN USER CUSTOMIZATION ===
mkdir -p checkpoint/
mkdir -p datapath/
mkdir -p dataset/dataset-config/text
mkdir -p model/

cd RWKV-v5 || exit

"${VENV_PYTHON}" ./init_model.py --n_layer 6 --n_embd 512 --vocab_size neox --skip-if-exists ../model/L6-D512-neox-init.pth
"${VENV_PYTHON}" ./init_model.py --n_layer 6 --n_embd 512 --vocab_size world --skip-if-exists ../model/L6-D512-world-init.pth

cd ../dataset/dataset-config/text || exit
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

cd ../../../RWKV-v5 || exit

"${VENV_PYTHON}" preload_datapath.py ../example-local-text.yaml
"${VENV_PYTHON}" lightning_trainer.py fit -c ../example-local-text.yaml
# === END USER CUSTOMIZATION ===
