#!/bin/bash

usage() {
  echo "Usage: $0 -fpk <fpk_file> -key <public_key_url> -hash <expected_hash>"
  echo "  -key, --public_key_url     URL to the public key in PEM format"
  echo "  -hash, --expected_hash     Expected SHA256 hash of the public key file"
  echo "  -fpk, --fpk_file           Path to the signed .fpk file"
  exit 1
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
fi

if ! command -v curl &>/dev/null; then
  echo "Error: curl is not installed."
  exit 2
fi

if ! command -v python3 &>/dev/null; then
  echo "Error: python3 is not installed."
  exit 3
fi

if [ "$#" -lt 6 ]; then
  usage
fi

while [[ "$#" -gt 0 ]]; do
  case $1 in
  -key | --public_key_url)
    PUBLIC_KEY_URL="$2"
    shift
    ;;
  -hash | --expected_hash)
    EXPECTED_HASH="$2"
    shift
    ;;
  -fpk | --fpk_file)
    FPK_FILE="$2"
    shift
    ;;
  *)
    echo "Error: Unknown parameter passed: $1"
    usage
    ;;
  esac
  shift
done

if [ -z "$PUBLIC_KEY_URL" ] || [ -z "$EXPECTED_HASH" ] || [ -z "$FPK_FILE" ]; then
  echo "Error: Missing required parameters."
  usage
fi

if [[ ! "$PUBLIC_KEY_URL" =~ ^https?:// ]]; then
  echo "Error: Invalid URL format."
  exit 5
fi

if [ ! -f "$FPK_FILE" ]; then
  echo "Error: The specified .fpk file does not exist."
  exit 4
fi

# Set up virtual environment using python3 -m venv
VENV_DIR=$(mktemp -d)
python3 -m venv "$VENV_DIR"

# Ensure that the virtual environment is deleted after the script finishes, no matter how it exits
trap 'deactivate; rm -rf "$VENV_DIR"' EXIT

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install required Python packages
pip install --quiet cryptography==42.0.7 zstandard==0.22.0

# Path to the Python script
PYTHON_SCRIPT="verify_signed_data.py"

PUBLIC_KEY_PATH=$(mktemp)

echo "Downloading public key from $PUBLIC_KEY_URL..."
if ! curl -s -o "$PUBLIC_KEY_PATH" "$PUBLIC_KEY_URL"; then
  echo "Error: Failed to download the public key."
  exit 6
fi

if [ ! -s "$PUBLIC_KEY_PATH" ]; then
  echo "Error: Public key file was not downloaded or is empty."
  exit 7
fi

echo "Verifying the public key hash..."
CALCULATED_HASH=$(sha256sum "$PUBLIC_KEY_PATH" | awk '{print $1}')

if [ "$CALCULATED_HASH" != "$EXPECTED_HASH" ]; then
  echo "Error: Public key hash does not match the expected value."
  exit 8
fi

echo "Verifying the signature of $FPK_FILE using Python script..."

# Call the Python script for verification
python3 "$PYTHON_SCRIPT" --public_key "$PUBLIC_KEY_PATH" --fpk "$FPK_FILE"
VERIFICATION_RESULT=$?

# Handle verification result
if [ $VERIFICATION_RESULT -eq 0 ]; then
  echo "The signature is valid."
else
  echo "The signature is NOT valid."
  exit 11
fi
