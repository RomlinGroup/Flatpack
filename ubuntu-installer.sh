#!/bin/bash
set -e
set -u

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

REQUIRED_PACKAGES="bc curl build-essential libssl-dev pkg-config apt-transport-https ca-certificates software-properties-common"
MISSING_PACKAGES=""

for pkg in $REQUIRED_PACKAGES; do
  if ! dpkg -l | grep -q "^ii  $pkg"; then
    MISSING_PACKAGES+="$pkg "
  fi
done

if [ -n "${MISSING_PACKAGES}" ]; then
  echo "🔎 The following packages are missing and will be installed: $MISSING_PACKAGES"
  prompt_continue
  sudo apt update
  sudo apt install -y "$MISSING_PACKAGES"
fi

if ! command -v docker &>/dev/null; then
  echo "🐳 Docker is not installed. Installing it now..."
  prompt_continue

  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

  if ! grep -q "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" /etc/apt/sources.list /etc/apt/sources.list.d/*; then
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt update
  fi

  if sudo apt install -y docker-ce docker-ce-cli containerd.io; then
    echo "✔️ Docker installed successfully."
  else
    echo "❗ Docker installation failed. Exiting."
    exit 1
  fi
else
  echo "🐳 Docker is already installed."
fi

if ! command -v rustc &>/dev/null; then
  echo "🦀 Rust is not installed. Installing it now..."
  prompt_continue
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  echo "source $HOME/.cargo/env" >>~/.bashrc
  # shellcheck disable=SC1090
  source ~/.bashrc

  if command -v rustc &>/dev/null; then
    echo "✔️ Rust installed successfully."
  else
    echo "❗ Rust installation failed. Exiting."
    exit 1
  fi
else
  echo "🦀 Rust is already installed."
fi

# Check the installation by printing the version
echo "🦀 Checking Rust installation..."
bash -c "source $HOME/.cargo/env; rustc --version"

echo "🐳 Checking Docker installation..."
bash -c "docker --version"

echo "✔️ All necessary tools are installed. You are ready to go!"
