#!/bin/bash

#init_script="$SCRIPT_DIR/init.sh"
#[ ! -f "$init_script" ] && curl -s "https://raw.githubusercontent.com/romlingroup/flatpack/main/warehouse/init.sh" -o "$init_script"
#source "$init_script"

init_script="$SCRIPT_DIR/init.sh"
remote_script_url="https://raw.githubusercontent.com/romlingroup/flatpack/main/warehouse/init.sh"
temp_remote_script=$(mktemp)
curl -s "$remote_script_url" -o "$temp_remote_script"
local_version=$(grep 'INIT_VERSION' "$init_script" 2>/dev/null | cut -d '"' -f 2 || echo "0.0.0")
remote_version=$(grep 'INIT_VERSION' "$temp_remote_script" | cut -d '"' -f 2)
echo "[LOCAL] $local_version"
echo "[REMOTE] $remote_version"

if [[ "$remote_version" > "$local_version" ]]; then
    mv "$temp_remote_script" "$init_script" && echo "✨ Updated to $remote_version."
else
    rm "$temp_remote_script"
fi

source "$init_script" 2>/dev/null || echo "❌ Failed to load init.sh."

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

build_python