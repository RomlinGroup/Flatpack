#!/bin/bash
set -e
set -u

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
  echo "❌ Python is not installed. Please install Python and try again."
  exit 1
fi

# Check if pip is installed
if ! command_exists pip; then
  echo "❌ pip is not installed. Please install pip and try again."
  exit 1
fi

# Create a directory to hold the virtual environment
echo "📁 Creating a directory to hold the virtual environment..."
mkdir myenv
cd myenv

# Install virtualenv if not already installed
echo "🔧 Installing virtualenv..."
python3 -m pip install --quiet virtualenv

# Create the virtual environment
echo "🚀 Creating the virtual environment..."
python3 -m virtualenv venv >/dev/null 2>&1

# Activate the virtual environment
echo "🌍 Activating the virtual environment..."
source venv/bin/activate >/dev/null 2>&1

# Install dependencies
echo "🧩 Installing dependencies..."
pip install --quiet torch numpy pandas scikit-learn matplotlib

# Run the Python script
echo "🐍 Running your Python script..."
python - <<EOF
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
EOF

# Deactivate the virtual environment
echo "🛑 Deactivating the virtual environment..."
env -i bash -c "source venv/bin/activate && deactivate" >/dev/null 2>&1

# Clean up
echo "🧹 Cleaning up..."
cd ..
rm -rf myenv

echo "🎉 Script completed successfully!"