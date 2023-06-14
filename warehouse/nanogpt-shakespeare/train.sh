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
else
  WORK_DIR="/content/nanogpt-gpt2/nanoGPT"
  DEVICE="cuda"
fi

cd "$WORK_DIR" || exit
python train.py config/train_shakespeare_char.py --device=$DEVICE --compile=False
