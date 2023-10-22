#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 train.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=edge-model
export FLATPACK_NAME=edge-model
# === END USER CUSTOMIZATION ===

source "$SCRIPT_DIR/device.sh" || {
  echo "⚠️ Error: Failed to source device.sh" >&2
  exit 1
}

# === BEGIN USER CUSTOMIZATION ===
#"${VENV_PYTHON}" -c "import time; [print(f'Count: {i}') or time.sleep(1) for i in range(10, 0, -1)]"
#echo "🎉 Happy New Year!"

source "$SCRIPT_DIR/$REPO_NAME/llama_cpp_test.sh" || {
  echo "⚠️ Error: Failed to source $REPO_NAME/llama_cpp_test.sh" >&2
  exit 1
}
# === END USER CUSTOMIZATION ===
