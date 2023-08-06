# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-math'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'math'
wandb_run_name = 'mini-gpt'

dataset = 'math'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 20 # attention window

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # good values are beween 1e-3 and 1e-4
max_iters = 10000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = learning_rate / 10
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'mps'
compile = False
