#!/bin/bash

# === BEGIN USER CUSTOMIZATION ===
REPO_NAME="GPT2-Knowledge-Distillation"
FLATPACK_NAME="gpt2-knowledge-distillation"
# === END USER CUSTOMIZATION ===

# Get the directory of the script
SCRIPT_DIR="$(dirname $0)"

# Ensure logs directory exists
mkdir -p "$SCRIPT_DIR/logs"

# Function to log with timestamps
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >>"$SCRIPT_DIR/logs/output.log"
}

# Check for required commands
for cmd in "python" "bash"; do
  command -v $cmd >/dev/null 2>&1 || {
    log "$cmd is required but it's not installed."
    exit 1
  }
done

log "Starting script..."

{
  # Source device script
  source "$SCRIPT_DIR/device.sh"

  # === BEGIN USER CUSTOMIZATION ===
  python "$SCRIPT_DIR/data/shakespeare/prepare.py"
  bash "$SCRIPT_DIR/run_adamw/train_student.sh"
  # === END USER CUSTOMIZATION ===

} >>"$SCRIPT_DIR/logs/output.log" 2>>"$SCRIPT_DIR/logs/error.log" || log "An error occurred!"
