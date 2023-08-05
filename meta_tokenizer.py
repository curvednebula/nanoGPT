import pickle

class MetaTokenizer:

    def __init__(self, meta_path):
        print(f"Loading vocab from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        print(f"Found vocab_size = {self.vocab_size} in {meta_path}")

    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
