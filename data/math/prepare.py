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

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

with open(input_file_path, 'w') as f:
    for i in range(1000):
        for j in range(1000):
            f.write(f"{i}+{j}={i+j}\n")

# download the dataset
with open(input_file_path, 'r') as f:
    data = f.read()

print(f"Length of dataset in characters: {len(data):,}")

tokenizer = Tokenizer(from_data=data)
tokenizer.to_file(os.path.join(os.path.dirname(__file__), 'meta.pkl'))

# create the train and validation splits - make sure that validation split starts after new-line
val_data_index = data.find('\n', int(len(data)*0.9)) + 1
train_data = data[:val_data_index]
val_data = data[val_data_index:]

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

