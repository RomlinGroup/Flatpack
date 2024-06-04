part_bash """
# Part 1: Bash
PRESENT=\"Part 1: Hello from the present!\"
echo \$PRESENT
PAST=\"Part 10: Hello from the past!\"
"""
part_bash """
# Part 2: Current Directory
echo \"Part 2: Current Directory:\"
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
# Part 4: Listing Files
echo \"Part 4: Listing Files:\"
ls -la
"""
part_python """
# Part 5: Python - Define Function
print(\"Part 5: Defining compute_average function\")

def compute_average(numbers):
    return sum(numbers) / len(numbers)
"""
part_python """
# Part 6: Python - Subprocess Example
print(\"Part 6: Running subprocess to check directory\")
result = subprocess.run(['pwd'], capture_output=True, text=True)
print(result.stdout.strip())
"""
part_bash """
# Part 7: Listing Files Again
echo \"Part 7: Listing Files Again:\"
ls -la
"""
part_python """
# Part 8: Python - Define Factorial Function
print(\"Part 8: Defining compute_factorial function\")

def compute_factorial(num):
    return math.factorial(num)
"""
part_python """
# Part 9: Python - Using Functions
print(\"Part 9: Factorial of 5 is:\", compute_factorial(5))
print(\"Part 9: The average is:\", compute_average(numbers))
"""
part_bash """
# Part 10: Bash
echo \$PAST
"""