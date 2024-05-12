#!/bin/bash

TEMP_PYTHON_SCRIPT=$(mktemp /tmp/python_script.XXXXXX.py)
VALIDATOR_SCRIPT=$(mktemp /tmp/validator_script.XXXXXX.py)

trap "rm -f $TEMP_PYTHON_SCRIPT $VALIDATOR_SCRIPT" EXIT

part_python() {
  echo "$1" >> "$TEMP_PYTHON_SCRIPT"
  echo "$1" >> "$VALIDATOR_SCRIPT"

  if ! "$VENV_PYTHON" -m py_compile "$VALIDATOR_SCRIPT"; then
    echo "❌ Invalid Python code after recent addition. Exiting..."
    exit 1
  fi
}

part_python """
import math
import subprocess
numbers = [10, 20, 30, 40, 50]
print('(1) Python starts here')
"""

part_python """
print('(2) def compute_average')
def compute_average(numbers):
  return sum(numbers) / len(numbers)
"""

part_python """
print('(3) subprocess.run')
subprocess.run(['pwd'])
"""

part_python """
print('(4) def compute_factorial')
def compute_factorial(num):
  return math.factorial(num)
"""

part_python """
print('(5) Factorial of 5 is:', compute_factorial(5))
print('(6) The average is:', compute_average(numbers))
"""

echo "✅ Python code is valid. Executing the script..."
"$VENV_PYTHON" "$TEMP_PYTHON_SCRIPT"