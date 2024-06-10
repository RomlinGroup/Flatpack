#!/bin/bash

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "[DEBUG] $(TZ=GMT date +"%Y-%m-%d %H:%M:%S")"
echo -e "🚀 build.sh is running in: $SCRIPT_DIR"

# === BEGIN USER CUSTOMIZATION ===
export DEFAULT_REPO_NAME=demo
export FLATPACK_NAME=demo
# === END USER CUSTOMIZATION ===

source "$SCRIPT_DIR/device.sh" || {
  echo "😱 Error: Failed to source device.sh" >&2
  exit 1
}

REQUIRED_DEVICES="cpu cuda mps"

if [[ ! $REQUIRED_DEVICES =~ (^|[[:space:]])$DEVICE($|[[:space:]]) ]]; then
  echo "😱 Error: This script requires one of the following devices: $REQUIRED_DEVICES." >&2
  exit 1
fi

# === BEGIN USER CUSTOMIZATION ===
init_script="$SCRIPT_DIR/init.sh"
[ -f "$init_script" ] || {
  echo "init.sh not found, exiting."
  exit 1
}
source "$init_script" || {
  echo "Failed to load init.sh."
  exit 1
}

source "$SCRIPT_DIR/custom.sh" || {
  echo "😱 Error: Failed to source custom.sh" >&2
  exit 1
}
# === END USER CUSTOMIZATION ===
