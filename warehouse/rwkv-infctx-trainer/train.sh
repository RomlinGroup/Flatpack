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
sed -i 's/max_steps: 10/max_steps: 100/' notebook/dataset-config/example-local-text.yaml

mkdir -p checkpoint/
mkdir -p datapath/
mkdir -p dataset/dataset-config/text/
mkdir -p model/

cd RWKV-v4neo || exit

"${VENV_PYTHON}" ./init_model.py --n_layer 6 --n_embd 512 --vocab_size neox --skip-if-exists ../model/L6-D512-neox-init.pth
"${VENV_PYTHON}" ./init_model.py --n_layer 6 --n_embd 512 --vocab_size world --skip-if-exists ../model/L6-D512-world-init.pth

cd ../text/ || exit
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

cd ../../../RWKV-v4neo || exit

"${VENV_PYTHON}" preload_datapath.py ../notebook/dataset-config/example-local-text.yaml
"${VENV_PYTHON}" lightning_trainer.py fit -c ../notebook/dataset-config/example-local-text.yaml
# === END USER CUSTOMIZATION ===
