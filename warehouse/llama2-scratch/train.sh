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
    WORK_DIR="llama2.c"
    DEVICE="mps"
  else
    WORK_DIR="/home/content/llama2.c"
    DEVICE="cpu"
  fi
else
  WORK_DIR="/content/llama2-scratch/llama2.c"
  DEVICE="cuda"
fi

cd "$WORK_DIR" || exit
python tinystories.py download
python tinystories.py pretokenize
python train.py
