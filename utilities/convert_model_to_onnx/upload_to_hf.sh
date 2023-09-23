#!/bin/bash
set -e
set -u

# Check if a token is already saved
if ! huggingface-cli whoami &>/dev/null; then
  # Prompt the user for the Hugging Face API token if not already logged in
  read -sp "Enter your Hugging Face API token: " HUGGINGFACE_TOKEN
  echo ""
  echo $HUGGINGFACE_TOKEN | huggingface-cli login
fi

# Check if the model name and model path are passed as command-line arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 model_name model_path"
  exit 1
fi

# Define model details
MODEL_NAME="$1"
MODEL_PATH="transformers.js/models/$2"
REPO_URL="https://huggingface.co/romlingroup/$MODEL_NAME"

# Create a new repository on Hugging Face
huggingface-cli repo create $MODEL_NAME --organization romlingroup

# Clone the repository, copy files, commit, and push
git clone $REPO_URL
cp -r $MODEL_PATH/* $MODEL_NAME/
cd $MODEL_NAME
git add .
git commit -m "Add model files"
git push

# Navigate back to the parent directory and remove the cloned repository folder
cd ..
rm -rf $MODEL_NAME

echo "Model uploaded to Hugging Face Model Hub: $REPO_URL"
