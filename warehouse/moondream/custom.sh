#"${VENV_PIP}" install -r requirements.txt
cp ../tiger.png tiger.png
"${VENV_PYTHON}" sample.py --image "tiger.png" --prompt "Should I pet this dog?"