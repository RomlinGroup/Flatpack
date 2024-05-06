#!/bin/bash

TEMP_PYTHON_SCRIPT=$(mktemp /tmp/python_script.XXXXXX.py)

trap "rm -f $TEMP_PYTHON_SCRIPT" EXIT

part_python() {
  cat <<EOF >> "$TEMP_PYTHON_SCRIPT"
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

"$VENV_PYTHON" "$TEMP_PYTHON_SCRIPT"