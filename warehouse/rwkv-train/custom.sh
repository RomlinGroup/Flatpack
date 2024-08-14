part_bash """
../bin/pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121

../bin/pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade

git clone https://github.com/BlinkDL/RWKV-LM

cd RWKV-LM/RWKV-v5

mkdir -p data

wget -nc --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx

wget -nc --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin

chmod +x demo-training-prepare.sh

chmod +x demo-training-run.sh

./demo-training-prepare.sh

./demo-training-run.sh
"""
