part_bash """
# Part 1: Bash
GREETING=\"Part 1: Hello, World!\"
<div>echo \$GREETING</div><div>
</div>ADIEU=\"Part 10: Adieu, Monde!\"
"""
part_bash """
# Part 2: Current directory
echo \"Part 2: Current directory:\"
pwd
"""
part_python """
# Part 3: Python
import math
import subprocess

numbers = [10, 20, 30, 40, 50]
print(\"Part 3: Python starts here\")
"""
part_bash """
# Part 4: Listing files
echo \"Part 4: Listing files:\"
ls -la
"""
part_python """
# Part 5: Python
print(\"Part 5: def compute_average\")

def compute_average(numbers):
  return sum(numbers) / len(numbers)
"""
part_python """
# Part 6: Python
print(\"Part 6: Running subprocess to check directory\")
<div><br></div><div>result = subprocess.run(['pwd'], capture_output=True, text=True)
</div>print(result.stdout.strip())
"""
part_bash """
# Part 7: Listing files again
echo \"Part 7: Listing files again:\"

ls -la
"""
part_python """
# Part 8: Python
<div>print(\"Part 8: def compute_factorial\")</div><div>
</div>def compute_factorial(num):
  return math.factorial(num)
"""
part_python """
# Part 9: Python
print(\"Part 9: Factorial of 5 is:\", compute_factorial(5))
print(\"Part 9: The average is:\", compute_average(numbers))
"""
part_bash """
# Part 10: Bash
echo \$ADIEU
"""
part_python """
<div># Part 11: Python<br></div><div>print(\"Part 11: Running subprocess to check directory\")</div><div>
</div><div>result = subprocess.run(['pwd'], capture_output=True, text=True)
</div><div>print(result.stdout.strip())</div>
"""