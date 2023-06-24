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
  WORK_DIR="/content/lit-parrot-redpajama/lit-gpt"
fi

cd "$WORK_DIR" || exit
pip install huggingface_hub
python scripts/download.py --repo_id stabilityai/stablelm-base-alpha-3b
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b

# Change the values of DATA_FILE_URL and DATA_FILE_NAME in prepare_alpaca.py
sed -i 's#DATA_FILE_URL = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"#DATA_FILE_URL = "https://raw.githubusercontent.com/romlingroup/OpenAlpaca/main/openalpaca.json"#' scripts/prepare_alpaca.py
sed -i 's#DATA_FILE_NAME = "alpaca_data_cleaned_archive.json"#DATA_FILE_NAME = "openalpaca.json"#' scripts/prepare_alpaca.py
python scripts/prepare_alpaca.py --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b

# Change the values of batch_size and micro_batch_size in lora.py
sed -i 's/batch_size = 128/batch_size = 64/' finetune/lora.py
sed -i 's/micro_batch_size = 4/micro_batch_size = 4/' finetune/lora.py
python finetune/lora.py --precision bf16-true --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
