#!/bin/bash

# Check if lsb_release is available
if ! command -v lsb_release &> /dev/null; then
  echo "❌ lsb_release command not found. This script requires lsb_release to check the OS."
  exit 1
fi

# Determine the OS
OS=$(lsb_release -is 2>/dev/null)

if [[ "$OS" == "Ubuntu" ]]; then
  echo "🐧 Setting up Ubuntu environment"

  # Add deadsnakes PPA for Python versions
  if ! sudo add-apt-repository ppa:deadsnakes/ppa -y; then
    echo "❌ Failed to add deadsnakes PPA."
    exit 1
  fi

  # Update package list
  if ! sudo apt-get update; then
    echo "❌ Failed to update package list."
    exit 1
  fi

  # Install necessary packages
  if ! sudo apt-get install -y build-essential git python3.11 python3.11-dev python3.11-venv python3-dev; then
    echo "❌ Failed to install necessary packages."
    exit 1
  fi

  # Install pipx
  if ! sudo apt-get install -y pipx; then
    echo "❌ Failed to install pipx."
    exit 1
  fi

  # Ensure pipx is in the PATH
  if ! pipx ensurepath; then
    echo "❌ Failed to ensure pipx is in PATH."
    exit 1
  fi

  # Add pipx to PATH for the current session
  export PATH="$PATH:~/.local/bin"

  # Source shell configuration if it exists
  if [ -f ~/.bashrc ]; then
    source ~/.bashrc
  elif [ -f ~/.zshrc ]; then
    source ~/.zshrc
  fi

  # Install flatpack using pipx with Python 3.11
  if ! pipx install flatpack --python python3.11; then
    echo "❌ Failed to install flatpack using pipx with Python 3.11."
    exit 1
  fi

  echo "✅ Setup completed successfully."
else
  echo "❌ The installer currently only supports Ubuntu."
  exit 1
fi