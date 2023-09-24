#!/bin/bash

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=GPT2-Knowledge-Distillation
export FLATPACK_NAME=gpt2-knowledge-distillation
# === END USER CUSTOMIZATION ===

source ./device.sh || { echo "Error: Failed to source device.sh" >&2; exit 1; }

# === BEGIN USER CUSTOMIZATION ===
cd data/shakespeare
python prepare.py
bash run_adamw/train_student.sh
# === END USER CUSTOMIZATION ===