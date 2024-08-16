#!/bin/bash

trap cleanup EXIT

cleanup() {
  deactivate &>/dev/null
  rm -rf verify_venv
  echo "Cleaned up virtual environment."
}

python3 -m venv verify_venv
source verify_venv/bin/activate

pip install cryptography==43.0.0 zstandard==0.23.0

echo "Virtual environment created, packages installed."
