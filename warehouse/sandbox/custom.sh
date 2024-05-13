#!/bin/bash

init_script="$SCRIPT_DIR/init.sh"
[ -f "$init_script" ] || { echo "init.sh not found, exiting."; exit 1; }
source "$init_script" || { echo "Failed to load init.sh."; exit 1; }

# DO NOT EDIT ABOVE THIS LINE

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

# DO NOT EDIT BELOW THIS LINE

build_python