#!/bin/bash

# "${VENV_PIP}"
# "${VENV_PYTHON}"

init_script="$SCRIPT_DIR/init.sh"
[ -f "$init_script" ] || { echo "init.sh not found, exiting."; exit 1; }
source "$init_script" || { echo "Failed to load init.sh."; exit 1; }

"${VENV_PIP}" install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

if [ -f ".build_successful" ]; then
    echo "✅ Build already completed for llama.cpp"
else
    echo "🦙 Building llama.cpp"
    if make; then
        echo "✅ Finished building llama.cpp"
        touch .build_successful
    else
        echo "❌ Build failed for llama.cpp"
        exit 1
    fi
fi