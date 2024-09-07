part_bash """
../bin/pip install pytorch-lightning==1.9.5
"""
part_bash """
git clone https://github.com/BlinkDL/RWKV-LM

cd RWKV-LM/RWKV-v5

mkdir -p data
"""
part_bash """
wget -nc -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
sudo apt-get -y install nvidia-cuda-toolkit
sudo apt-get -y install ninja-build

wget -nc -q --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx

wget -nc -q --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin

../../../bin/pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
../../../bin/pip install deepspeed wandb ninja --upgrade
"""
part_bash """
# Prepare training

../../../bin/python train.py --proj_dir \"out/L12-D768-x060\" --data_file \"data/minipile\" --data_type \"binidx\" --vocab_size 65536 --my_testing x060 --ctx_len 512 --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 --epoch_save 1 --weight_decay 0 --head_size_a 64 --num_nodes 1 --micro_bsz 1 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 --my_exit_tokens 1498226207 --magic_prime 2926181 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1

# Run training

../../../bin/python train.py --load_model \"0\" --proj_dir \"out/L12-D768-x060\" --my_testing x060 --ctx_len 512 --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 --data_file \"data/minipile\" --my_exit_tokens 1498226207 --magic_prime 2926181 --num_nodes 1 --micro_bsz 16 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 --lr_init 6e-4 --lr_final 6e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --data_type \"binidx\" --vocab_size 65536 --weight_decay 0.001 --epoch_save 10 --head_size_a 64 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 --enable_progress_bar True --ds_bucket_mb 2
"""
