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

if ! command -v openssl &>/dev/null; then
  echo "Error: openssl is not installed."
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

TEMP_DATA_FILE=$(mktemp)
TEMP_SIGNATURE_FILE=$(mktemp)

trap 'rm -f "$PUBLIC_KEY_PATH" "$TEMP_DATA_FILE" "$TEMP_SIGNATURE_FILE"' EXIT

separator="---SIGNATURE_SEPARATOR---"

awk -v RS="$separator" -v data="$TEMP_DATA_FILE" -v sig="$TEMP_SIGNATURE_FILE" 'NR==1{print > data} NR==2{print > sig}' "$FPK_FILE"

if [ ! -s "$TEMP_DATA_FILE" ]; then
  echo "Error: Data file is empty or was not created."
  exit 9
fi

if [ ! -s "$TEMP_SIGNATURE_FILE" ]; then
  echo "Error: Signature file is empty or was not created."
  exit 10
fi

echo "Verifying the signature of $FPK_FILE..."

if openssl dgst -sha256 -sigopt rsa_padding_mode:pss -sigopt rsa_mgf1_md:sha256 -sigopt rsa_pss_saltlen:-1 \
  -verify "$PUBLIC_KEY_PATH" -signature "$TEMP_SIGNATURE_FILE" "$TEMP_DATA_FILE"; then
  echo "The signature is valid."
else
  echo "The signature is NOT valid."
  exit 11
fi
