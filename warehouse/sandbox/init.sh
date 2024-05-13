#!/bin/bash

echo "📦 Initializing the FPK package"

TEMP_PYTHON_SCRIPT="/tmp/python_script.py"
VALIDATOR_SCRIPT="/tmp/validator_script.py"

rm -f "$TEMP_PYTHON_SCRIPT" "$VALIDATOR_SCRIPT"

touch "$TEMP_PYTHON_SCRIPT" "$VALIDATOR_SCRIPT"
trap "rm -f '$TEMP_PYTHON_SCRIPT' '$VALIDATOR_SCRIPT'" EXIT

# build_python
build_python() {
    if [ -f "$TEMP_PYTHON_SCRIPT" ]; then
        echo "✅ Python code is valid. Building the script..."
        "$VENV_PYTHON" "$TEMP_PYTHON_SCRIPT"
    else
        echo "❌ Python script file does not exist."
    fi
}

# part_python
part_python() {
  echo "$1" >> "$TEMP_PYTHON_SCRIPT"
  echo "$1" >> "$VALIDATOR_SCRIPT"

  if ! "$VENV_PYTHON" -m py_compile "$VALIDATOR_SCRIPT"; then
    echo "❌ Invalid Python code after recent addition. Exiting..."
    exit 1
  fi
}