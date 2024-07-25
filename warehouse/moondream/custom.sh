part_bash """
if [ -f ../tiger.png ]; then
    cp -f ../tiger.png tiger.png
fi
"""
part_bash """
../bin/python sample.py --image \"tiger.png\" --cpu --prompt \"Should I pet this dog?\"
"""