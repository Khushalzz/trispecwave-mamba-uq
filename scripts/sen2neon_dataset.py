import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio as rio
import numpy as np
import random
import pandas as pd

class SEN2NEONDataset(Dataset):
    def __init__(self, root_dir, csv_path, split="train", crop_size_lr=64):
        self.root_dir = Path(root_dir)
        self.crop_size_lr = crop_size_lr
        
        df = pd.read_csv(csv_path)
        
        # Determine train/val splits by hash or just split the dataframe deterministically
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * 0.9) # 90% train, 10% val
        
        if split == "train":
            self.df = df.iloc[:split_idx]
        else:
            self.df = df.iloc[split_idx:]
            
        self.lr_paths = [self.root_dir / p for p in self.df["lr_path" if "lr_path" in self.df.columns else "lr"]]
        self.hr_paths = [self.root_dir / p for p in self.df["hr_2_5m_path" if "hr_2_5m_path" in self.df.columns else "hr"]]

    def __len__(self):
        return len(self.df)

    def _read_tiff(self, path):
        with rio.open(path) as ds:
            arr = ds.read()
            nodata = ds.nodata
            
        arr = arr.astype(np.float32)
        if nodata is not None:
            arr = np.where(arr == nodata, 0.0, arr)
            
        return arr

    def __getitem__(self, idx):
        lr = self._read_tiff(self.lr_paths[idx])
        hr = self._read_tiff(self.hr_paths[idx])

        # Align crops
        s = hr.shape[1] // lr.shape[1] # Scale factor (e.g. 1024 // 256 = 4)
        
        if self.crop_size_lr:
            H, W = lr.shape[1], lr.shape[2]
            y = random.randint(0, H - self.crop_size_lr)
            x = random.randint(0, W - self.crop_size_lr)
            
            lr = lr[:, y:y+self.crop_size_lr, x:x+self.crop_size_lr]
            hr = hr[:, y*s:(y+self.crop_size_lr)*s, x*s:(x+self.crop_size_lr)*s]

        # Convert to [0, 1] range based on 10,000 scale
        lr = torch.from_numpy(lr) / 10000.0
        hr = torch.from_numpy(hr) / 10000.0
        
        return torch.nan_to_num(lr).clamp(0, 1), torch.nan_to_num(hr).clamp(0, 1)
