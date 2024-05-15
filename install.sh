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

  if ! sudo apt-get install -y build-essential python3.11 python3.11-dev python3.11-venv; then
    echo "❌ Failed to install necessary packages."
    exit 1
  fi

  if ! curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py; then
    echo "❌ Failed to download get-pip.py."
    exit 1
  fi

  if ! sudo python3.11 get-pip.py; then
    echo "❌ Failed to install pip for Python 3.11."
    exit 1
  fi

  rm get-pip.py

  if ! sudo python3.11 -m pip install flatpack --no-cache-dir; then
    echo "❌ Failed to install flatpack."
    exit 1
  fi

  echo "✅ Setup completed successfully."
else
  echo "❌ The installer currently only supports Ubuntu."
  exit 1
fi