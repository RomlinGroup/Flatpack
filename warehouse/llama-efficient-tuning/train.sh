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
    WORK_DIR="LLaMA-Efficient-Tuning"
    DEVICE="mps"
  else
    WORK_DIR="/home/content/LLaMA-Efficient-Tuning"
    DEVICE="cpu"
  fi
else
  WORK_DIR="/content/llama-efficient-tuning/LLaMA-Efficient-Tuning"
  DEVICE="cuda"
fi

cd "$WORK_DIR" || exit
python train_model.py