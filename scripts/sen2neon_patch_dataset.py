import random
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ChunkedPatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, chunk_dir, shuffle=True, augment=False):
        self.chunk_paths = sorted(list(Path(chunk_dir).glob("chunk_*.pt")))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No chunk_*.pt files found in {chunk_dir}")
        self.shuffle = shuffle
        self.augment = augment
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Determine which chunks this worker is responsible for
        if worker_info is None:
            paths = self.chunk_paths.copy()
        else:
            # Split chunks across workers
            paths = [
                p for i, p in enumerate(self.chunk_paths) 
                if i % worker_info.num_workers == worker_info.id
            ]
            
        if self.shuffle:
            random.shuffle(paths)
            
        for path in paths:
            data = torch.load(path, weights_only=True)
            
            # Convert chunk from uint16 physical values to float32 [0,1]
            lr_chunk = (data["lr"].float() / 10000.0).clamp(0, 1)
            hr_chunk = (data["hr"].float() / 10000.0).clamp(0, 1)
            
            n = len(lr_chunk)
            indices = list(range(n))
            if self.shuffle:
                random.shuffle(indices)
                
            for i in indices:
                lr = lr_chunk[i]
                hr = hr_chunk[i]
                
                if self.augment:
                    if random.random() > 0.5:
                        lr = torch.flip(lr, [-1])
                        hr = torch.flip(hr, [-1])
                    if random.random() > 0.5:
                        lr = torch.flip(lr, [-2])
                        hr = torch.flip(hr, [-2])
                        
                yield lr, hr
