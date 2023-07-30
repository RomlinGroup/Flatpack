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
  WORK_DIR="/content/lit-gpt-falcon/lit-gpt"
fi

cd "$WORK_DIR" || exit
python scripts/download.py --repo_id tiiuae/falcon-7b
python scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/tiiuae/falcon-7b

python scripts/prepare_alpaca.py \
  --destination_path data/alpaca \
  --checkpoint_dir checkpoints/tiiuae/falcon-7b

python finetune/adapter_v2.py \
  --data_dir data/alpaca \
  --checkpoint_dir checkpoints/tiiuae/falcon-7b \
  --out_dir out/adapter/alpaca \
  --precision 32-true
