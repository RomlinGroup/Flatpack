#!/bin/bash

# Get the directory where the script is located
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "🚀 server.sh is running in: $SCRIPT_DIR\n"

# === BEGIN USER CUSTOMIZATION ===
export REPO_NAME=llama.cpp
export FLATPACK_NAME=obsidian-multi-modal
# === END USER CUSTOMIZATION ===

source "$SCRIPT_DIR/device.sh" || {
  echo "😱 Error: Failed to source device.sh" >&2
  exit 1
}

# Required devices (cpu cuda mps)
REQUIRED_DEVICES="cuda mps"

# Check if DEVICE is among the required devices
if [[ ! " $REQUIRED_DEVICES " =~ " $DEVICE " ]]; then
  echo "😱 Error: This script requires one of the following devices: $REQUIRED_DEVICES." >&2
  exit 1
fi

# === BEGIN USER CUSTOMIZATION ===
echo "Checking if 'server' executable exists..."
if [ ! -f "server" ]; then
    echo "'server' not found. Starting compilation..."
    make
    if [ $? -eq 0 ]; then
        echo "'server' compiled successfully."
    else
        echo "Compilation failed. Please check the makefile and source code for errors."
        exit 1
    fi
else
    echo "'server' executable already exists. Skipping compilation."
fi

echo "Ensuring 'models' directory exists..."
if [[ ! -d "models" ]]; then
    mkdir -p "models"
    echo "'models' directory created."
else
    echo "'models' directory already exists. Moving on..."
fi

echo "Checking for model 'mmproj-obsidian-f16.gguf'..."
if [ ! -f models/mmproj-obsidian-f16.gguf ]; then
    echo "Model 'mmproj-obsidian-f16.gguf' not found. Downloading..."
    wget -O models/mmproj-obsidian-f16.gguf https://huggingface.co/NousResearch/Obsidian-3B-V0.5-GGUF/resolve/main/mmproj-obsidian-f16.gguf
    echo "Download complete."
else
    echo "Model 'mmproj-obsidian-f16.gguf' already exists, skipping download."
fi

echo "Checking for model 'obsidian-q6.gguf'..."
if [ ! -f models/obsidian-q6.gguf ]; then
    echo "Model 'obsidian-q6.gguf' not found. Downloading..."
    wget -O models/obsidian-q6.gguf https://huggingface.co/NousResearch/Obsidian-3B-V0.5-GGUF/resolve/main/obsidian-q6.gguf
    echo "Download complete."
else
    echo "Model 'obsidian-q6.gguf' already exists, skipping download."
fi

echo "Attempting to run 'server' with specified models..."
./server -m models/obsidian-q6.gguf --mmproj models/mmproj-obsidian-f16.gguf -ngl 42
if [ $? -eq 0 ]; then
    echo "'server' started successfully."
else
    echo "'server' failed to start. Please check the command line arguments and model files."
    exit 2
fi
# === END USER CUSTOMIZATION ===