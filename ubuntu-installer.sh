#!/bin/bash

# Check if bc and curl are installed
if ! command -v bc &> /dev/null || ! command -v curl &> /dev/null; then
    echo "This script requires bc and curl. You need to install them manually."
    exit 1
fi

# Get the Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)

# Check if the version is less than 20.04
if (( $(echo "$UBUNTU_VERSION < 20.04" | bc -l) )); then
    echo "This script requires Ubuntu 20.04 or higher. Exiting."
    exit 1
fi

# Check if Rust is already installed
if ! command -v rustc &> /dev/null; then
    # Download and install Rust via rustup
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    # Add Rust to PATH manually
    echo 'source $HOME/.cargo/env' >> ~/.bashrc
else
    echo "Rust is already installed. Skipping installation."
fi

# Check the installation by printing the version
bash -c "source $HOME/.cargo/env; rustc --version"