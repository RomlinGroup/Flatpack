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
    WORK_DIR="RWKV-LM"
    DEVICE="mps"
  else
    WORK_DIR="/home/content/RWKV-LM"
    DEVICE="cpu"
  fi
else
  WORK_DIR="/content/rwkv-lm-finetuning/RWKV-LM"
  DEVICE="cuda"
fi

cd "$WORK_DIR" || exit
