#!/bin/bash
set -e

python nanogpt_prepare.py

cd nanoChatGPT
python train.py config/scratch-gpt2.py