#!/bin/bash
set -e

# Save the original directory
original_dir=$(pwd)

# Check if the nanoGPT directory exists
if [ -d "nanoChatGPT" ]; then
    cd nanoChatGPT
else
    chmod +x nanogpt_setup.sh
    ./nanogpt_setup.sh
fi

python nanogpt_prepare.py