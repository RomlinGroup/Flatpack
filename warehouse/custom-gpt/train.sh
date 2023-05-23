#!/bin/sh
cd /home/content/nanoGPT || exit
echo "🚀 Starting training..."
python3 train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
status=$?
if [ $status -eq 0 ]; then
  echo "✅ Training completed successfully!"
else
  echo "❌ Training failed. Please check the logs above for details."
fi
