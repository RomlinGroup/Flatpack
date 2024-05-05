"${VENV_PIP}" install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

cp train.py train.py.backup
sed -i '' "s/device = 'cuda'/device = 'mps'/" train.py
sed -i '' 's/compile = True/compile = False/' train.py
"${VENV_PYTHON}" data/shakespeare_char/prepare.py
"${VENV_PYTHON}" train.py config/train_shakespeare_char.py