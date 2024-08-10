part_bash """
if [ -f ../tiger.png ]; then
    cp -f ../tiger.png tiger.png
fi
"""
part_bash """
../bin/python sample.py \
--cpu \
--image \"tiger.png\" \
--prompt \"Should I pet this dog?\" \
> output.txt \
2>log.txt
"""
part_bash """
if [ -f \"output.txt\" ]; then
    echo \"Running hooks for output.txt:\"
    cat output.txt
else
    echo \"Error: output.txt does not exist.\"
fi
"""