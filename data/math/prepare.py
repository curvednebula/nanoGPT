"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Need a better way to import from parent dir
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(os.path.join(parent_dir, 'tokenizer'))
from tokenizer import Tokenizer

train_file_path = os.path.join(os.path.dirname(__file__), 'train.txt')
with open(train_file_path, 'w') as f:
    # for i in range(0, 101, 1):
    #     for j in range(0, 101, 1):
    #         f.write(f"{i}+{j}={i+j}\n")
    for i in range(0, 1000, 7):
        for j in range(0, 1000, 11):
            f.write(f"{i}+{j}={i+j}\n") 

val_file_path = os.path.join(os.path.dirname(__file__), 'val.txt')
with open(val_file_path, 'w') as f:
    # for i in range(20, 30, 1):
    #     for j in range(70, 80, 1):
    #         f.write(f"{i}+{j}={i+j}\n")
    for i in range(200, 300, 2):
        for j in range(700, 800, 3):
            f.write(f"{i}+{j}={i+j}\n")
    # for i in range(2000, 2500, 1):
    #     for j in range(7000, 7500, 1):
    #         f.write(f"{i}+{j}={i+j}\n")

# download the dataset
with open(train_file_path, 'r') as f:
    train_data = f.read()

with open(val_file_path, 'r') as f:
    val_data = f.read()

print(f"Length of train dataset: {len(train_data)}")
print(f"Length of train dataset: {len(val_data)}")

tokenizer = Tokenizer(from_data=train_data)
tokenizer.to_file(os.path.join(os.path.dirname(__file__), 'meta.pkl'))

# encode both to integers
train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

