#!/bin/bash
set -e
set -u

echo_stage() {
  printf "\n\033[1;34m>>> %s\033[0m\n" "$1"
}

echo_success() {
  printf "\033[1;32m✓ %s\033[0m\n" "$1"
}

echo_failure() {
  printf "\033[1;31m✗ Error: %s\033[0m\n" "$1"
  exit 1
}

echo_stage "Checking for pipx installation"
if ! command -v pipx &>/dev/null; then
  echo "Pipx not found. Attempting to install..."
  if command -v brew &>/dev/null; then
    echo "Using Homebrew to install pipx..."
    brew install pipx || echo_failure "Homebrew pipx installation failed"
  elif command -v apt-get &>/dev/null; then
    echo "Using apt-get to install pipx..."
    sudo apt-get update
    sudo apt-get install -y pipx || echo_failure "apt-get pipx installation failed"
  elif command -v pip3 &>/dev/null; then
    echo "Using pip3 to install pipx..."
    pip3 install --user pipx || echo_failure "pip3 pipx installation failed"
  else
    echo_failure "No supported package manager found to install pipx"
  fi

  echo "Configuring pipx path..."
  pipx ensurepath || echo_failure "Adding pipx to PATH"

  echo "Enabling global pipx usage..."
  sudo pipx ensurepath --global || echo_failure "Enabling global pipx"

  echo_success "Pipx installed successfully"
else
  echo_success "Pipx is already installed"
fi

echo_stage "Uninstalling existing Flatpack (if exists)"
if pipx list | grep -q flatpack; then
  pipx uninstall flatpack
  echo_success "Existing Flatpack uninstalled"
else
  echo "No existing Flatpack installation found"
fi

echo_stage "Installing Flatpack"
pipx install flatpack || echo_failure "Flatpack installation failed"
echo_success "Flatpack installed successfully"

echo_stage "Cleaning up test folder"
if [ -d "test" ]; then
  rm -rf test
  echo_success "Deleted existing test folder"
else
  echo "No existing test folder found"
fi

echo_stage "Verifying flatpack list"
output=$(flatpack list)
if [[ "$output" == *"Directory Name"* ]]; then
  echo_success "Flatpack list command executed successfully"
else
  echo_failure "Flatpack list execution failed"
fi

echo_stage "Unboxing test directory"
echo "YES" | flatpack unbox test
echo_success "Flatpack unbox test completed"

echo_stage "Checking test folder structure"
if [[ -d "test" && -d "test/build" && $(ls -A "test/build") ]]; then
  echo_success "Test folder and build directory verified"
else
  echo_failure "Test folder or build directory is missing or empty"
fi

echo_stage "Executing flatpack build test"
if flatpack build test &>/dev/null; then
  echo_success "Flatpack build test executed successfully"

  if [ -f "test/build/test/test_pass" ]; then
    echo_success "Test pass file found - Verification complete"
  else
    echo_failure "Test pass file not found"
  fi
else
  echo_failure "Flatpack build test execution failed"
fi

printf "\n\033[1;32m🎉 All stages completed successfully!\033[0m\n"
