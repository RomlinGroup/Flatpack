#!/bin/bash

if ! command -v lsb_release &> /dev/null; then
  echo "❌ lsb_release command not found. This script requires lsb_release to check the OS."
  exit 1
fi

OS=$(lsb_release -is 2>/dev/null)

if [[ "$OS" == "Ubuntu" ]]; then
  echo "🐧 Setting up Ubuntu environment"

  if ! sudo add-apt-repository ppa:deadsnakes/ppa -y; then
    echo "❌ Failed to add deadsnakes PPA."
    exit 1
  fi

  if ! sudo apt-get update; then
    echo "❌ Failed to update package list."
    exit 1
  fi

  if ! sudo apt-get install -y build-essential python3-dev python3.11 python3.11-dev python3.11-venv; then
    echo "❌ Failed to install necessary packages."
    exit 1
  fi

  if ! sudo apt-get install -y pipx; then
    echo "❌ Failed to install pipx."
    exit 1
  fi

  if ! pipx ensurepath; then
    echo "❌ Failed to ensure pipx is in PATH."
    exit 1
  fi

  export PATH="$PATH:~/.local/bin"

  if [ -f ~/.bashrc ]; then
    source ~/.bashrc
  elif [ -f ~/.zshrc ]; then
    source ~/.zshrc
  fi

  echo "✅ Setup completed successfully."
else
  echo "❌ The installer currently only supports Ubuntu."
  exit 1
fi
