#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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

  # Running the scripts
  python "$SCRIPT_DIR/data/shakespeare/prepare.py"
  bash "$SCRIPT_DIR/run_adamw/train_student.sh"

} >>"$SCRIPT_DIR/logs/output.log" 2>>"$SCRIPT_DIR/logs/error.log" || log "An error occurred!"
