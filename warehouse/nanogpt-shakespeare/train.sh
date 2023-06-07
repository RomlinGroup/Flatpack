#!/bin/sh

OS=$(uname)
if [ "$OS" = "Darwin" ]; then
  DEVICE="mps"
else
  DEVICE="cpu"
fi

python train.py config/train_shakespeare_char.py --device=$DEVICE --compile=False
