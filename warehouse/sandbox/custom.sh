#!/bin/bash

TEMP_PYTHON_SCRIPT=$(mktemp /tmp/python_script.XXXXXX.py)
VALIDATOR_SCRIPT=$(mktemp /tmp/validator_script.XXXXXX.py)

trap "rm -f $TEMP_PYTHON_SCRIPT $VALIDATOR_SCRIPT" EXIT

part_python() {
  cat <<EOF >> "$TEMP_PYTHON_SCRIPT"
$1
EOF

  cat <<EOF >> "$VALIDATOR_SCRIPT"
$1
EOF
}

part_python """
print(\"Python script execution started...\")

def compute_average(numbers):
  return sum(numbers) / len(numbers)
"""

part_python """
import math

def compute_factorial(num):
  return math.factorial(num)

print('Factorial of 5 is:', compute_factorial(5))
"""

part_python """
numbers = [10, 20, 30, 40, 50]
print('The average is:', compute_average(numbers))
"""

if python3 -m py_compile "$VALIDATOR_SCRIPT"; then
  echo "✅ Python code is valid. Executing the script..."
  "$VENV_PYTHON" "$TEMP_PYTHON_SCRIPT"
else
  echo "❌ Invalid Python code. Exiting..."
  exit 1
fi