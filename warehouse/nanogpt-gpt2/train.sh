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
    WORK_DIR="nanoGPT"
    DEVICE="mps"
  else
    WORK_DIR="/home/content/nanoGPT"
    DEVICE="cpu"
  fi

  cd "$WORK_DIR" || exit
else
  cd "/content/nanogpt-gpt2/nanoGPT" || exit
fi

python train.py config/finetune_shakespeare.py --device=$DEVICE --compile=False
