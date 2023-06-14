import time
out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())
dataset = 'shakespeare'
init_from = 'gpt2-large'
always_save_checkpoint = False
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 1000
learning_rate = 1e-6
decay_lr = True
warmup_iters = 200
lr_decay_iters = max_iters
min_lr = learning_rate/10