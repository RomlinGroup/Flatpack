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
numbers = [10, 20, 30, 40, 50]
"""

part_python """
def compute_average(numbers):
  return sum(numbers) / len(numbers)
"""

part_python """
def compute_factorial(num):
  return math.factorial(num)
"""

part_python """
print('Factorial of 5 is:', compute_factorial(5))
print('The average is:', compute_average(numbers))
"""

echo "✅ Python code is valid. Executing the script..."
"$VENV_PYTHON" "$TEMP_PYTHON_SCRIPT"
