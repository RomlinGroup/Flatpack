#!/bin/bash

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

python train.py config/train_shakespeare_char.py --device=$DEVICE --compile=False
