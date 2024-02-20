#!/bin/bash
set -e

# Save the original directory
original_dir=$(pwd)

# Check if the nanoGPT directory exists
if [ -d "nanoGPT" ]; then
    # If it exists, cd into it
    cd nanoGPT
else
    # If it doesn't exist, run setup.sh
    # Ensure setup.sh is executable or add chmod +x setup.sh before running it
    ./setup.sh
fi

# Assuming prepare.py and train.py are relative to the nanoGPT directory
python data/shakespeare_char/prepare.py

python train.py \
  --dataset=shakespeare_char \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=64 \
  --compile=False \
  --eval_iters=1 \
  --block_size=64 \
  --batch_size=8 \
  --device=mps