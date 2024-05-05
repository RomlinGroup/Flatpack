#!/bin/bash

if [[ -z "$VENV_PYTHON" ]]; then
    echo "Error: The VENV_PYTHON environment variable is not set."
    exit 1
fi

TEMP_PYTHON_SCRIPT=$(mktemp /tmp/python_script.XXXXXX.py)

trap "rm -f $TEMP_PYTHON_SCRIPT" EXIT

cat << EOF > $TEMP_PYTHON_SCRIPT
print("Python script execution started...")
def compute_average(numbers):
    return sum(numbers) / len(numbers)
EOF

cat << EOF >> $TEMP_PYTHON_SCRIPT

# Another block: compute factorial
import math
def compute_factorial(num):
    return math.factorial(num)
print("Factorial of 5 is:", compute_factorial(5))

EOF

cat << EOF >> $TEMP_PYTHON_SCRIPT

# Further usage of previous definitions
numbers = [10, 20, 30, 40, 50]
print("The average is:", compute_average(numbers))
EOF

"$VENV_PYTHON" $TEMP_PYTHON_SCRIPT
