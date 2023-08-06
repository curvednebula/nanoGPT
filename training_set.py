import os
import numpy as np
import torch

class TrainingSet:

    def __init__(self, data_dir: str, device: str, device_type: str, block_size: int, batch_size: int):
        self.device = device
        self.device_type = device_type
        self.block_size = block_size
        self.batch_size = batch_size

        with open(os.path.join(data_dir, 'train.bin'), 'rb') as file:
            self.train_data = np.frombuffer(file.read(), dtype=np.int16)
        with open(os.path.join(data_dir, 'val.bin'), 'rb') as file:
            self.val_data = np.frombuffer(file.read(), dtype=np.int16)

        self.qna_delimeter = None
        self.train_data_qna = None
        self.val_data_qna = None

    def split_into_qna_sets(self, delimeter: int):
        self.qna_delimeter = delimeter
        self.train_data_qna = self._split(self.train_data, delimeter)
        self.val_data_qna = self._split(self.val_data, delimeter)

    def _split(self, data: np.ndarray, delimiter: int) -> list[np.ndarray]:
        # Find the indices where the delimiter occurs
        delimiter_indices = np.where(data == delimiter)[0]
        start_index = 0
        subarrays: list[np.ndarray] = []

        for i in delimiter_indices:
            subarray = data[start_index:i]
            if (subarray.size > 0):
                subarrays.append(subarray)
            start_index = i + 1

        # Convert the list of subarrays back to a numpy array if needed
        # subarrays = [np.array(subarray) for subarray in subarrays]
        return subarrays

    # split: 'train' | 'val'
    def get_batch(self, split: str):
        if self.train_data_qna and self.val_data_qna:
            x, y = self._get_qna_batch(split)
        else:
            x, y = self._get_random_batch(split)

        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        
        return x, y
    
    def _get_qna_batch(self, split: str):
        
        data = self.train_data_qna if split == 'train' else self.val_data_qna
        ix = torch.randint(len(data), (self.batch_size,))
        
        xbatches: list[torch.Tensor] = []
        ybatches: list[torch.Tensor] = []

        for rnd in ix:
            qna: np.ndarray = data[rnd]
            xpadding = np.full(self.block_size - qna.size, self.qna_delimeter, qna.dtype)
            xbatch = np.concatenate((qna, xpadding)).astype(np.int64)
            ypadding = np.full(self.block_size - qna.size + 1, self.qna_delimeter, qna.dtype)
            ybatch = np.concatenate((qna[1:], ypadding)).astype(np.int64)
            xbatches.append(torch.from_numpy(xbatch))
            ybatches.append(torch.from_numpy(ybatch))
        
        x = torch.stack(xbatches)
        y = torch.stack(ybatches)
        return x, y

    def _get_random_batch(self, split: str):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        return x, y