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
python src/train_bash.py \
    --stage pt \
    --model_name_or_path tiiuae/falcon-7b-instruct \
    --do_train \
    --dataset wiki_demo \
    --template default \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir /content/output \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 100000 \
    --plot_loss \
    --fp16