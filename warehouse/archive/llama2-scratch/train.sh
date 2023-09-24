#!/bin/bash

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=llama2.c
export FLATPACK_NAME=llama2-scratch
# === END USER CUSTOMIZATION ===

source ./device.sh || { echo "Error: Failed to source device.sh" >&2; exit 1; }

# === BEGIN USER CUSTOMIZATION ===
python tinystories.py download
python tinystories.py pretokenize
python train.py
# === END USER CUSTOMIZATION ===