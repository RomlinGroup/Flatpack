#!/bin/bash
set -e

# Save the original directory
original_dir=$(pwd)

# Check if the nanoGPT directory exists
if [ -d "nanoChatGPT" ]; then
    cd nanoChatGPT
else
    chmod +x setup.sh
    ./setup.sh
fi

python prepare.py