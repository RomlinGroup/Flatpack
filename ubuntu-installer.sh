#!/bin/bash

# Check if bc and curl are installed
if ! command -v bc &> /dev/null || ! command -v curl &> /dev/null; then
    echo "This script requires bc and curl. Installing them now..."
    sudo apt update
    sudo apt install -y bc curl
fi

# Get the Ubuntu version
UBUNTU_VERSION=$(dpkg-query --showformat='${Version}' --show base-files | cut -d '.' -f 1,2)

# Check if the version is less than 20.04
if (( $(echo "$UBUNTU_VERSION < 20.04" | bc -l) )); then
    echo "This script requires Ubuntu 20.04 or higher. Exiting."
    exit 1
fi

# Update the package lists for upgrades and new package installations
sudo apt update

# Check if Rust is already installed
if ! command -v rustc &> /dev/null; then
    # Download and install Rust via rustup
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    # Add Rust to PATH manually
    echo 'source $HOME/.cargo/env' >> ~/.bashrc
    source ~/.bashrc
else
    echo "Rust is already installed. Skipping installation."
fi

# Check the installation by printing the version
rustc --version