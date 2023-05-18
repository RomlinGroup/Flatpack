#!/bin/bash

# Prompt the user before proceeding with each major step
function prompt_continue {
  read -rp "Continue? (y/n) " choice
  case "$choice" in
  y | Y) echo "😊 Proceeding..." ;;
  n | N)
    echo "👋 Exiting script."
    exit 1
    ;;
  *)
    echo "❗ Invalid input."
    prompt_continue
    ;;
  esac
}

echo "🔎 Checking if bc and curl are installed..."

# Check if bc and curl are installed
if ! command -v bc &>/dev/null || ! command -v curl &>/dev/null; then
  echo "❗ This script requires bc and curl. Installing them now..."
  prompt_continue
  sudo apt update
  sudo apt install -y bc curl
else
  echo "✔️ bc and curl are installed."
fi

echo "🔎 Checking Ubuntu version..."

# Get the Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)

# Check if the version is less than 20.04
if (($(echo "$UBUNTU_VERSION < 20.04" | bc -l))); then
  echo "❗ This script requires Ubuntu 20.04 or higher. Exiting."
  exit 1
else
  echo "✔️ Ubuntu version is $UBUNTU_VERSION, which is supported."
fi

echo "🔎 Checking if C compiler (usually gcc) is installed..."

# Check if cc is installed (cc is usually a symlink to gcc)
if ! command -v cc &>/dev/null; then
  echo "❗ This script requires a C compiler (usually gcc). Installing it now..."
  prompt_continue
  sudo apt update
  sudo apt install -y build-essential
else
  echo "✔️ C compiler is installed."
fi

echo "🔎 Checking if OpenSSL and pkg-config are installed..."

# Check if OpenSSL and pkg-config are installed
if ! command -v openssl &>/dev/null || ! command -v pkg-config &>/dev/null; then
  echo "❗ This script requires OpenSSL and pkg-config. Installing them now..."
  prompt_continue
  sudo apt update
  sudo apt install -y libssl-dev pkg-config
else
  echo "✔️ OpenSSL and pkg-config are installed."
fi

echo "🔎 Checking if Rust is installed..."

# Check if Rust is already installed
if ! command -v rustc &>/dev/null; then
  echo "🦀 Rust is not installed. Installing it now..."
  prompt_continue
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  # Add Rust to PATH manually
  echo "source $HOME/.cargo/env" >>~/.bashrc
  # shellcheck disable=SC1090
  source ~/.bashrc
else
  echo "🦀 Rust is already installed. Skipping installation."
fi

# Check the installation by printing the version
echo "🦀 Checking Rust installation..."
bash -c "source $HOME/.cargo/env; rustc --version"
