#!/bin/bash
set -e
set -x

# Check for necessary tools
if ! command -v git &> /dev/null; then
    echo "git could not be found, please install it first."
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "cmake could not be found, please install it first."
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo "make could not be found, please install it first."
    exit 1
fi

# Clone or update llama.cpp repository
if [ -d "llama.cpp" ]; then
    echo "Directory 'llama.cpp' already exists. Updating repository."
    cd llama.cpp
    git pull
else
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
fi

# Prepare the build directory
if [ -d "build" ]; then
    echo "Build directory already exists. Cleaning up."
    rm -rf build
fi
mkdir build
cd build

# Configure and compile
cmake .. -DCMAKE_APPLE_SILICON_PROCESSOR=arm64
make -j