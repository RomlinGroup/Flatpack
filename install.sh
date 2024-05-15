#!/bin/bash

# Check if lsb_release is installed
if ! command -v lsb_release &> /dev/null; then
  echo "❌ lsb_release command not found. This script requires lsb_release to check the OS."
  exit 1
fi

# Get the operating system name
OS=$(lsb_release -is 2>/dev/null)

# Check if the operating system is Ubuntu
if [[ "$OS" == "Ubuntu" ]]; then
  echo "🐧 Setting up Ubuntu environment"

  # Add deadsnakes PPA
  if ! sudo add-apt-repository ppa:deadsnakes/ppa -y; then
    echo "❌ Failed to add deadsnakes PPA."
    exit 1
  fi

  # Update package list
  if ! sudo apt-get update; then
    echo "❌ Failed to update package list."
    exit 1
  fi

  # Install build-essential, Python 3.11, and related packages
  if ! sudo apt-get install -y build-essential python3.11 python3.11-dev python3.11-venv; then
    echo "❌ Failed to install necessary packages."
    exit 1
  fi

  # Install pipx
  if ! sudo apt-get install -y pipx; then
    echo "❌ Failed to install pipx."
    exit 1
  fi

  # Ensure pipx is in PATH
  if ! pipx ensurepath; then
    echo "❌ Failed to ensure pipx is in PATH."
    exit 1
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