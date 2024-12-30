#!/bin/bash
set -e
set -u

if [[ "$(uname)" != "Darwin" ]]; then
  echo "This script is designed to run only on macOS."
  exit 1
fi

echo_stage() {
  echo "Stage: $1"
}

echo_failure() {
  echo "Failure in stage: $1"
  exit 1
}

echo_stage "Checking for pipx..."
if ! command -v pipx &>/dev/null; then
  echo "pipx not found, installing..."
  brew install pipx || echo_failure "Installing pipx"
  echo "Ensuring pipx is in your PATH..."
  pipx ensurepath || echo_failure "Adding pipx to PATH"
  echo "Ensuring pipx can be used with --global (requires sudo)..."
  sudo pipx ensurepath --global || echo_failure "Enabling pipx --global"
else
  echo "pipx already installed."
fi

# echo_stage "Uninstalling flatpack (if it exists) using pipx..."
# if pipx uninstall flatpack &>/dev/null; then
#   echo "Uninstalled flatpack."
# else
#   echo "flatpack not found, skipping uninstall..."
# fi

# echo_stage "Installing flatpack using pipx..."
# pipx install flatpack || echo_failure "Installing flatpack"

echo_stage "Testing flatpack list..."
output=$(flatpack list)
if [[ "$output" == *"Directory Name"* ]]; then
  echo "flatpack list executed successfully."
else
  echo_failure "flatpack list execution"
fi

echo_stage "Testing flatpack unbox llama-cpp..."
echo "YES" | flatpack unbox llama-cpp
echo "flatpack unbox llama-cpp executed successfully."

echo_stage "Checking for llama-cpp folder and build..."
if [[ -d "llama-cpp" && -d "llama-cpp/build" && $(ls -A "llama-cpp/build") ]]; then
  echo "llama-cpp folder and non-empty build directory exist."
else
  echo_failure "llama-cpp folder or build directory not found or empty"
fi

# echo_stage "Deleting llama-cpp (if it exists)..."
# if flatpack delete llama-cpp &>/dev/null; then
#   echo "Deleted llama-cpp."
# else
#   echo "llama-cpp not found, skipping delete..."
# fi

echo "All stages completed successfully."
