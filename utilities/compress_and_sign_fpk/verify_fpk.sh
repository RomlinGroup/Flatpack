#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <fpk_path>"
  exit 1
fi

if ! command -v python3 &>/dev/null; then
  echo "Python 3 is required but not installed."
  exit 1
fi

if ! python3 -m pip --version &>/dev/null; then
  echo "pip for Python 3 is required but not installed."
  exit 1
fi

fpk_path="$1"
temp_dir="temp_verify"
original_dir=$(pwd)

cleanup() {
  cd "$original_dir" || exit 1
  rm -rf "$temp_dir"
}

trap cleanup EXIT

mkdir -p "$temp_dir"
cp "$fpk_path" "$temp_dir/"
cd "$temp_dir" || exit 1

python3 -m venv verify_venv
source verify_venv/bin/activate

if ! python3 -c "import cryptography" &>/dev/null || ! python3 -c "import zstandard" &>/dev/null; then
  pip install cryptography==43.0.0 zstandard==0.23.0
fi

download_file() {
  url="$1"
  output="$2"
  if command -v wget &>/dev/null; then
    wget -O "$output" "$url"
  elif command -v curl &>/dev/null; then
    curl -o "$output" "$url"
  else
    echo "Neither wget nor curl is available for downloading files."
    exit 1
  fi
}

download_file "https://raw.githubusercontent.com/RomlinGroup/Flatpack/main/utilities/compress_and_sign_fpk/verify_signed_data_with_cli.py" "verify_signed_data_with_cli.py"
download_file "https://raw.githubusercontent.com/RomlinGroup/Flatpack/main/public_key.pem" "public_key.pem"

python verify_signed_data_with_cli.py --public_key public_key.pem --f "$(basename "$fpk_path")"

exit 0
