#!/bin/bash

# "${VENV_PIP}"
# "${VENV_PYTHON}"

init_script="$SCRIPT_DIR/init.sh"
[ -f "$init_script" ] || { echo "init.sh not found, exiting."; exit 1; }
source "$init_script" || { echo "Failed to load init.sh."; exit 1; }

"${VENV_PIP}" install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

cp train.py train.py.backup

SED_CMD="sed -i"
[[ "$OS" == "Darwin" ]] && SED_CMD="sed -i ''"
$SED_CMD "s/device = 'cuda'/device = '$( [[ "$OS" == "Darwin" ]] && echo 'mps' || echo 'cpu' )'/" train.py
$SED_CMD 's/compile = True/compile = False/' train.py

"${VENV_PYTHON}" data/shakespeare_char/prepare.py
"${VENV_PYTHON}" train.py config/train_shakespeare_char.py