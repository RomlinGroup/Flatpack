import time
out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())
dataset = 'shakespeare'
init_from = 'gpt2'
always_save_checkpoint = True
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 1000
learning_rate = 3e-5
decay_lr = False