#!/bin/bash

# "${VENV_PIP}"
# "${VENV_PYTHON}"

init_script="$SCRIPT_DIR/init.sh"
[ -f "$init_script" ] || { echo "init.sh not found, exiting."; exit 1; }
source "$init_script" || { echo "Failed to load init.sh."; exit 1; }

"${VENV_PIP}" install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

cp train.py train.py.backup
sed -i '' "s/device = 'cuda'/device = 'mps'/" train.py
sed -i '' 's/compile = True/compile = False/' train.py
"${VENV_PYTHON}" data/shakespeare_char/prepare.py
"${VENV_PYTHON}" train.py config/train_shakespeare_char.py