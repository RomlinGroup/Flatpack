#!/bin/bash

# === BEGIN USER CUSTOMIZATION ===
REPO_NAME="GPT2-Knowledge-Distillation"
FLATPACK_NAME="gpt2-knowledge-distillation"
# === END USER CUSTOMIZATION ===

# Ensure logs directory exists
mkdir -p logs

# Function to log with timestamps
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >>./logs/output.log
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
  source ./device.sh

  # === BEGIN USER CUSTOMIZATION ===
  python data/shakespeare/prepare.py
  bash run_adamw/train_student.sh
  # === END USER CUSTOMIZATION ===

} >>./logs/output.log 2>>./logs/error.log || log "An error occurred!"
