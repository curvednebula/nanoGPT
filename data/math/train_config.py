dataset = 'math'
out_dir = 'out-math'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'math'
wandb_run_name = 'mini-gpt'

# GPT model parameters
block_size = 24 # attention window
n_layer = 6
n_head = 12
n_embd = n_head * 32
dropout = 0.2

# Learning parameters
gradient_accumulation_steps = 1
batch_size = 64

learning_rate = 1e-3 # good values are beween 1e-3 and 1e-6
min_lr = learning_rate / 10

max_iters = 10000
warmup_iters = 100 # not super necessary potentially
lr_decay_iters = int(max_iters * 0.9) # make equal to max_iters usually

beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# on macbook also add
device = 'mps'
compile = False
