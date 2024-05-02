#!/bin/bash

# Function to remove a directory
remove_dir() {
  local dir_name="$1"
  rm -rf "$dir_name" 2>/dev/null
  if [ $? -eq 0 ]; then
    echo "✔️  Successfully removed $dir_name!"
  else
    echo "ℹ️  $dir_name doesn't exist or couldn't be removed."
  fi
}

# 🚀 Starting package build script for PyPI

# 🗑️ Clear old build directories and files 🗑️
echo "🗑️  Cleaning up old build and distribution directories..."
remove_dir "build"
remove_dir "dist"
remove_dir "flatpack.egg-info"

# Ensure Python is installed
if ! command -v python3 &>/dev/null; then
  echo "❌ Python3 is not found. Please install it first."
  exit 1
fi

# 📦 Building the package
echo "🛠️  Building the PyPI package..."
python3 setup.py sdist bdist_wheel
if [ $? -eq 0 ]; then
  echo "🎉 Successfully built the PyPI package!"
else
  echo "❌ Failed to build the PyPI package. Please check for errors above."
fi

# Uninstalling existing flatpack package without confirmation
echo "🗑️  Uninstalling any existing version of flatpack..."
pip uninstall flatpack -y

# Installing the locally built version
echo "⚙️  Installing the locally built version of flatpack..."
pip install dist/*whl

if [ $? -eq 0 ]; then
  echo "🎉 Successfully installed the local version of flatpack!"
else
  echo "❌ Failed to install the local version of flatpack. Please check for errors above."
fi
