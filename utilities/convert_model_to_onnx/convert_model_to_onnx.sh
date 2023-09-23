#!/bin/bash
set -e
set -u

# Check if the model_id is passed as a command-line argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 model_id"
  exit 1
fi

MODEL_ID="$1"

echo -e "\n🚀 Starting the process..."

# Check if git is installed
if ! command -v git &>/dev/null; then
  echo -e "❌ Git could not be found. Please install it and try again."
  exit 1
fi

# Check if pip is installed
if ! command -v pip &>/dev/null; then
  echo -e "❌ Pip could not be found. Please install it and try again."
  exit 1
fi

# Check if Python is installed
if ! command -v python &>/dev/null; then
  echo -e "❌ Python could not be found. Please install it and try again."
  exit 1
fi

# Clone the repository
echo -e "\n🔍 Cloning the transformers.js repository..."
if git clone -b frozen-2023-09-23 --single-branch https://github.com/romlingroup/transformers.js.git; then
  echo -e "✅ Repository cloned successfully!"
  rm -rf transformers.js/.gitignore
  rm -rf transformers.js/.github
  rm -rf transformers.js/.git
else
  echo -e "❌ Failed to clone the repository."
  exit 1
fi

# Navigate to the repository directory
cd transformers.js/scripts

# Install requirements
echo -e "\n🔍 Installing requirements..."
if pip install -r requirements.txt; then
  echo -e "✅ Requirements installed successfully!"
else
  echo -e "❌ Failed to install requirements."
  exit 1
fi

cd ..

# Run the custom script
echo -e "\n🔍 Running the custom script..."
if python -m scripts.convert --quantize --model_id "$MODEL_ID"; then
  echo -e "✅ Custom script ran successfully!\n🎉 Process completed!"
else
  echo -e "❌ Failed to run the custom script."
  exit 1
fi
