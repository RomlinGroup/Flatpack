#!/bin/sh

# Set the working directory
WORK_DIR="/home/content/nanoGPT"
cd "$WORK_DIR" || exit

# Define configuration variables
TRAIN_SCRIPT="train.py"
TRAIN_CONFIG="config/train_shakespeare_char.py"
DEVICE="cpu"
COMPILE_FLAG="--compile=False"
EVAL_ITERS="--eval_iters=20"
LOG_INTERVAL="--log_interval=1"
BLOCK_SIZE="--block_size=64"
BATCH_SIZE="--batch_size=12"
N_LAYER="--n_layer=4"
N_HEAD="--n_head=4"
N_EMBD="--n_embd=128"
MAX_ITERS="--max_iters=2000"
LR_DECAY_ITERS="--lr_decay_iters=2000"
DROPOUT="--dropout=0.0"

# Logging function
log_info() {
  echo "$1"
}

# Error handling function
handle_error() {
  log_info "$1"
  exit 1
}

log_info "🚀 Starting training..."

# Run the training script
python3 "$TRAIN_SCRIPT" "$TRAIN_CONFIG" \
  --device="$DEVICE" \
  "$COMPILE_FLAG" \
  "$EVAL_ITERS" \
  "$LOG_INTERVAL" \
  "$BLOCK_SIZE" \
  "$BATCH_SIZE" \
  "$N_LAYER" \
  "$N_HEAD" \
  "$N_EMBD" \
  "$MAX_ITERS" \
  "$LR_DECAY_ITERS" \
  "$DROPOUT" || handle_error "❌ Training failed. Please check the logs above for details."

# Check the exit status
if [ $? -eq 0 ]; then
  log_info "✅ Training completed successfully!"
else
  handle_error "❌ Training failed. Please check the logs above for details."
fi
