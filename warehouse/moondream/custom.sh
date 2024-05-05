#!/bin/bash

# "${VENV_PIP}"
# "${VENV_PYTHON}"

cp ../tiger.png tiger.png
"${VENV_PYTHON}" sample.py --image "tiger.png" --prompt "Should I pet this dog?"