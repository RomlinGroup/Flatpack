#!/bin/bash

if [[ "${COLAB_GPU}" == "1" ]]; then
  echo "Running in Google Colab environment"
  IS_COLAB=1
else
  echo "Not running in Google Colab environment"
  IS_COLAB=0
fi

if [[ $IS_COLAB -eq 0 ]]; then
  OS=$(uname)
  if [ "$OS" = "Darwin" ]; then
    WORK_DIR="lit-gpt"
  else
    WORK_DIR="/home/content/lit-gpt"
  fi
else
  WORK_DIR="/content/lit-gpt-pythia/lit-gpt"
fi

cd "$WORK_DIR" || exit
python scripts/download.py --repo_id EleutherAI/pythia-410m
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/EleutherAI/pythia-410m

# Change the values of DATA_FILE_URL and DATA_FILE_NAME in prepare_alpaca.py
sed -i 's#DATA_FILE_URL = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"#DATA_FILE_URL = "https://raw.githubusercontent.com/romlingroup/OpenAlpaca/main/openalpaca.json"#' scripts/prepare_alpaca.py
sed -i 's#DATA_FILE_NAME = "alpaca_data_cleaned_archive.json"#DATA_FILE_NAME = "openalpaca.json"#' scripts/prepare_alpaca.py
python scripts/prepare_alpaca.py --checkpoint_dir checkpoints/EleutherAI/pythia-410m
python finetune/adapter.py --checkpoint_dir checkpoints/EleutherAI/pythia-410m --precision 32-true
