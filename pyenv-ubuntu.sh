#!/bin/bash
set -e
set -u

# Header comment
echo -e "🚀 Let's get pyenv set up on Ubuntu 23.04! 🚀"

# Check if user has sudo permissions
if ! sudo -v; then
  echo -e "😟 Oops! This script requires sudo permissions. Please run as a user with sudo access. 🛑"
  exit 1
fi

# Update package lists
echo -e "🔄 Updating package lists..."
sudo apt-get update

# Check if git is installed, if not, install it
if ! command -v git &>/dev/null; then
  echo -e "📦 Installing git..."
  sudo apt-get install -y git
else
  echo -e "✅ git is already installed. Moving on!"
fi

# Install required packages
echo -e "🔧 Installing required packages..."
sudo apt-get install -y --no-install-recommends \
  make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
  xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv if not already installed
if [ ! -d "$HOME/.pyenv" ]; then
  echo -e "📥 Downloading and installing pyenv..."
  git clone https://github.com/pyenv/pyenv.git ~/.pyenv
else
  echo -e "🎉 pyenv is already installed. Skipping the installation!"
fi

# Update .bashrc and .profile for pyenv setup
echo -e "📝 Configuring pyenv in .bashrc and .profile..."

# Configuration for .bashrc
cat <<EOL >>~/.bashrc
if shopt -q login_shell; then
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
fi
if command -v pyenv >/dev/null; then
    eval "$(pyenv init -)"
fi
EOL

# Configuration for .profile
cat <<EOL >>~/.profile
if [ -z "$BASH_VERSION" ]; then
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
fi
EOL

echo -e "🎉 Configured pyenv in .bashrc and .profile. To start using pyenv, restart your shell or run 'source ~/.bashrc'."
echo -e "🎊 Script completed successfully! Happy coding! 🎊"
