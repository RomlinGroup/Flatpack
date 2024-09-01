#!/bin/bash
set -e

# Function to remove a directory
remove_dir() {
  local dir_name="$1"
  rm -rf "$dir_name" 2>/dev/null && echo "✔️ Successfully removed $dir_name!" || echo "ℹ️ $dir_name doesn't exist or couldn't be removed."
}

# Function to check flatpack installation
check_flatpack() {
  echo "🔍 Checking flatpack installation..."
  which flatpack || echo "flatpack not found in PATH"
  if command -v pipx &>/dev/null; then
    pipx list | grep flatpack || echo "flatpack not installed via pipx"
  else
    echo "pipx not installed"
  fi
}

# 🚀 Starting package build script for PyPI
echo "🗑️ Cleaning up old build and distribution directories..."
remove_dir "build"
remove_dir "dist"
remove_dir "flatpack.egg-info"

# Check initial state
check_flatpack

# Ensure Python is installed and print version
command -v python3 >/dev/null 2>&1 || {
  echo "❌ Python3 is not found. Please install it first."
  exit 1
}
echo "🐍 Using Python version: $(python3 --version)"

# Ensure pipx is installed
if ! command -v pipx &>/dev/null; then
  echo "❌ pipx is not found. Installing it now..."
  python3 -m pip install --user pipx
  python3 -m pipx ensurepath
  echo "✔️ pipx installed successfully!"
fi

# Uninstall flatpack from pipx
echo "🗑️ Attempting to uninstall flatpack from pipx..."
pipx uninstall flatpack || true

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Ensure build dependencies are installed in the virtual environment
echo "📦 Ensuring build dependencies are installed..."
pip install --upgrade pip setuptools wheel build

# 📦 Building the package
echo "🛠️ Building the PyPI package..."
python -m build

# Deactivate the virtual environment
deactivate

# Installing the locally built version
echo "⚙️ Installing the locally built version of flatpack..."
pipx install dist/*.whl

# Ensure pipx executables are in PATH
echo "🔧 Ensuring pipx executables are in PATH..."
pipx ensurepath

echo "🎉 Installation complete. Checking final state..."
check_flatpack

echo "👉 You may need to restart your terminal or run 'source ~/.zshrc' (or equivalent) for PATH changes to take effect."