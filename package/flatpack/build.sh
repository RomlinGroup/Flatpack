#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

check_flatpack() {
  echo "Checking flatpack installation..."
  which flatpack || echo "flatpack not found in PATH"
  if command -v pipx &>/dev/null; then
    pipx list | grep flatpack || echo "flatpack not installed via pipx"
  else
    echo "pipx not installed"
  fi
}

remove_dir() {
  local dir_name="$1"
  rm -rf "$dir_name" 2>/dev/null && echo "Successfully removed $dir_name!" || echo "$dir_name doesn't exist or couldn't be removed."
}

echo "Cleaning up old build and distribution directories..."
remove_dir "$SCRIPT_DIR/build"
remove_dir "$SCRIPT_DIR/dist"
remove_dir "$SCRIPT_DIR/flatpack.egg-info"

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
  export PATH="$PATH:$(python3 -m site --user-base)/bin"
  echo "pipx installed successfully!"
fi

echo "Attempting to uninstall flatpack from pipx..."
pipx uninstall flatpack || true

echo "Setting up a virtual environment..."
python3 -m venv --prompt flatpack_env "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"

echo "Ensuring build dependencies are installed..."
python3 -m pip install --upgrade pip build setuptools wheel

echo "Building the PyPI package..."
python3 -m build

deactivate

pipx install --python python3.12 dist/*.whl
