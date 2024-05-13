#!/bin/bash

init_script="$SCRIPT_DIR/init.sh"
temp_remote_script=$(mktemp)

curl -s -o "$temp_remote_script" -H "Cache-Control: no-cache" -H "Pragma: no-cache" "https://raw.githubusercontent.com/romlingroup/flatpack/main/warehouse/init.sh?$(date +%s%N)"
local_version=$(grep 'INIT_VERSION' "$init_script" | cut -d '"' -f 2 || echo "0.0.0")
remote_version=$(grep 'INIT_VERSION' "$temp_remote_script" | cut -d '"' -f 2)

if [[ "$remote_version" > "$local_version" ]]; then
    mv "$temp_remote_script" "$init_script"
    echo "Updated to $remote_version."
else
    rm "$temp_remote_script"
fi

source "$init_script" || echo "Failed to load init.sh."

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