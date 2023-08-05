import pickle

class Tokenizer:

    def __init__(self, from_file: str = None, from_data: str = None):
        if from_file:
            print(f"Loading vocab from {from_file}...")
            with open(from_file, 'rb') as f:
                meta = pickle.load(f)
            self.vocab_size = meta['vocab_size']
            self.stoi = meta['stoi']
            self.itos = meta['itos']
        elif from_data:
            tokens = from_data
            # get all the unique characters that occur in this text
            vocab = sorted(list(set(tokens)))
            print("Vocab from data:", ' '.join(vocab))
            self.vocab_size = len(vocab)
            self.stoi = { s:i for i,s in enumerate(vocab) }
            self.itos = { i:s for i,s in enumerate(vocab) }

        print(f"Vocab size: {self.vocab_size}")

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]
    
    def decode(self, l) -> str:
        return ''.join([self.itos[i] for i in l])
    
    def to_file(self, file_path: str):
        # save vocab to file
        meta = {
            'vocab_size': self.vocab_size,
            'itos': self.itos,
            'stoi': self.stoi,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(meta, f)
