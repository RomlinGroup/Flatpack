#!/bin/bash
set -e

# Function to remove a directory
remove_dir() {
  local dir_name="$1"
  rm -rf "$dir_name" 2>/dev/null && echo "✔️ Successfully removed $dir_name!" || echo "ℹ️ $dir_name doesn't exist or couldn't be removed."
}

# 🚀 Starting package build script for PyPI
echo "🗑️ Cleaning up old build and distribution directories..."
remove_dir "build"
remove_dir "dist"
remove_dir "flatpack.egg-info"

# Ensure Python is installed and print version
command -v python3 >/dev/null 2>&1 || {
  echo "❌ Python3 is not found. Please install it first."
  exit 1
}
echo "🐍 Using Python version: $(python3 --version)"

# Ensure Homebrew is installed
command -v brew >/dev/null 2>&1 || {
  echo "❌ Homebrew is not found. Installing it now..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
}

# Check if flatpack is installed via Homebrew and uninstall if it is
if brew list flatpack &>/dev/null; then
  echo "🍺 Uninstalling Homebrew version of flatpack..."
  brew uninstall flatpack
fi

# Ensure pipx is installed using Homebrew
if ! command -v pipx &>/dev/null; then
  echo "❌ pipx is not found. Installing it now using Homebrew..."
  brew install pipx
  pipx ensurepath
  echo "✔️ pipx installed successfully!"
fi

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

# Uninstalling existing flatpack package without confirmation
echo "🗑️ Uninstalling any existing version of flatpack..."
pipx uninstall flatpack 2>/dev/null || true

# Installing the locally built version
echo "⚙️ Installing the locally built version of flatpack..."
pipx install dist/*.whl

# Ensure pipx executables are in PATH
echo "🔧 Ensuring pipx executables are in PATH..."
pipx ensurepath

echo "🎉 Successfully installed the local version of flatpack!"
echo "👉 You may need to restart your terminal or run 'source ~/.zshrc' (or equivalent) for PATH changes to take effect."
echo "📌 $(flatpack version)"
echo "🐍 Using Python version for pipx: $(pipx --version | grep "Python version" | cut -d':' -f2 | xargs)"
