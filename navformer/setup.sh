#!/bin/bash
set -e

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install pre-release versions of torch, torchvision, and torchaudio
echo "Installing pre-release versions of torch, torchvision, and torchaudio..."
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Clone the nanoChatGPT repository
echo "Cloning the nanoChatGPT repository..."
git clone https://github.com/romlingroup/nanoChatGPT

rm -r nanoChatGPT/data/Chat

echo "Setup completed successfully."
