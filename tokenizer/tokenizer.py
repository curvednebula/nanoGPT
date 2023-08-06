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
        
        self.decode_ignored: list[str] = []

        print(f"Vocab size: {self.vocab_size}")
    
    def ignore_on_decode(self, ignored: str):
        self.decode_ignored.append(ignored)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]
    
    def decode(self, l) -> str:
        decoded_str = ''
        for i in l:
            token = self.itos[i]
            if token not in self.decode_ignored:
                decoded_str = decoded_str + token
        return decoded_str
    
    def to_file(self, file_path: str):
        # save vocab to file
        meta = {
            'vocab_size': self.vocab_size,
            'itos': self.itos,
            'stoi': self.stoi,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(meta, f)
