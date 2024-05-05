#!/bin/bash

# "${VENV_PIP}"
# "${VENV_PYTHON}"

PYTHON_SCRIPT=$(mktemp /tmp/python_script.XXXXXX.py)

trap "rm -f $PYTHON_SCRIPT" EXIT

cat << EOF > $PYTHON_SCRIPT
print("Python script execution started...")
def compute_average(numbers):
    return sum(numbers) / len(numbers)

# Example usage
numbers = [10, 20, 30, 40, 50]
print("The average is:", compute_average(numbers))
EOF

"${VENV_PYTHON}" $PYTHON_SCRIPT