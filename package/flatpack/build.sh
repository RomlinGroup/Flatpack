#!/bin/bash
set -e

remove_dir() {
  local dir_name="$1"
  rm -rf "$dir_name" 2>/dev/null && echo "Successfully removed $dir_name!" || echo "$dir_name doesn't exist or couldn't be removed."
}

check_flatpack() {
  echo "Checking flatpack installation..."
  which flatpack || echo "flatpack not found in PATH"
  if command -v pipx &>/dev/null; then
    pipx list | grep flatpack || echo "flatpack not installed via pipx"
  else
    echo "pipx not installed"
  fi
}

echo "Cleaning up old build and distribution directories..."

remove_dir "build"
remove_dir "dist"
remove_dir "flatpack.egg-info"

check_flatpack

command -v python3 >/dev/null 2>&1 || {
  echo "Python3 is not found. Please install it first."
  exit 1
}

echo "Using Python version: $(python3 --version)"

if ! command -v pipx &>/dev/null; then
  echo "pipx is not found. Installing it now..."
  python3 -m pip install --user pipx
  python3 -m pipx ensurepath
  echo "pipx installed successfully!"
fi

echo "Attempting to uninstall flatpack from pipx..."
pipx uninstall flatpack || true

python3 -m venv venv
source venv/bin/activate

echo "Ensuring build dependencies are installed..."
pip install --upgrade pip setuptools wheel build

echo "Building the PyPI package..."
python -m build

deactivate
