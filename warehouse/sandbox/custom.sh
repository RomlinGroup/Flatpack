#!/bin/bash

# "${VENV_PIP}"
# "${VENV_PYTHON}"

if [[ -z "$VENV_PYTHON" ]]; then
    echo "Error: The VENV_PYTHON environment variable is not set."
    exit 1
fi

TEMP_PYTHON_SCRIPT=$(mktemp /tmp/python_script.XXXXXX.py)

trap "rm -f $TEMP_PYTHON_SCRIPT" EXIT

python_block() {
    cat <<EOF >> "$TEMP_PYTHON_SCRIPT"
$1
EOF
}

python_block """print(\"Python script execution started...\")
def compute_average(numbers):
    return sum(numbers) / len(numbers)"""

python_block """
# Another block: compute factorial
import math
def compute_factorial(num):
    return math.factorial(num)
print('Factorial of 5 is:', compute_factorial(5))"""

python_block """
# Further usage of previous definitions
numbers = [10, 20, 30, 40, 50]
print('The average is:', compute_average(numbers))"""

"$VENV_PYTHON" "$TEMP_PYTHON_SCRIPT"
