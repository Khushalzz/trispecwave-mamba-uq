import os
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import psutil
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class SEN2NEONRamDataset(Dataset):
    """
    LR tiles: cached in shared RAM as uint16 (lossless, ~6.6 GB).
    HR tiles: windowed read directly from NVMe /tmp (float32).
    """
    def __init__(self, root_dir, csv_path, split="train",
                 crop_size_lr=None, scale_factor=4, max_workers=8):
        self.root_dir = Path(root_dir)
        self.crop_size_lr = crop_size_lr
        self.scale_factor = scale_factor

        df = pd.read_csv(csv_path)
        
        # Determine column names dynamically
        lr_col = next((c for c in df.columns if "lr" in c.lower()), None)
        hr_col = next((c for c in df.columns if "hr" in c.lower() and "2_5" in c.lower()), None) \
               or next((c for c in df.columns if "hr" in c.lower()), None)
        if not lr_col or not hr_col:
            raise ValueError(f"Missing LR/HR columns. Found: {list(df.columns)}")

        # For grouping/splitting (optional, fallback to random)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * 0.9)
        self.df = (df.iloc[:split_idx] if split == "train" else df.iloc[split_idx:]).reset_index(drop=True)

        self.lr_paths = [self.root_dir / str(p).strip("/") for p in self.df[lr_col]]
        self.hr_paths = [self.root_dir / str(p).strip("/") for p in self.df[hr_col]]
        n = len(self.df)

        logger.info(f"[{split.upper()}] Found {n} samples.")
        
        # LR RAM Cache (uint16)
        est_lr_bytes = n * 12 * 256 * 256 * 2 # uint16
        avail = psutil.virtual_memory().available
        self.lr_cache = None
        self.lr_in_ram = False

        if avail > est_lr_bytes + 4 * 1024**3:
            try:
                logger.info(f"Preloading {n} LR tiles into shared RAM ({est_lr_bytes/1e9:.2f} GB) as uint16...")
                # Allocate non-pinned, shared-memory tensor (pinned happens per-batch in DataLoader)
                self.lr_cache = torch.empty((n, 12, 256, 256), dtype=torch.int16)

                def _load_lr(i):
                    with rio.open(self.lr_paths[i]) as ds:
                        arr = ds.read()
                        nd = ds.nodata
                    if nd is not None:
                        arr = np.where(arr == nd, 0, arr)
                    # Store as int16/uint16 directly from rasterio to save memory
                    self.lr_cache[i].copy_(torch.from_numpy(arr.astype(np.int16)))

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    list(ex.map(_load_lr, range(n)))
                self.lr_in_ram = True
                logger.info("LR cache complete.")
            except Exception as e:
                logger.warning(f"RAM preload failed: {e}. Falling back to disk.")
                self.lr_cache = None
        else:
            logger.warning(f"Not enough RAM for LR cache. Available: {avail/1e9:.2f} GB")

    def __len__(self):
        return len(self.df)

    def _read_window(self, path, x, y, w, h):
        with rio.open(path) as ds:
            arr = ds.read(window=Window(x, y, w, h))
            nd = ds.nodata
        if nd is not None:
            arr = np.where(arr == nd, 0, arr)
        # Convert physical uint16 to float32 [0, 1]
        return torch.from_numpy(arr.astype(np.float32) / 10000.0).clamp(0, 1)

    def __getitem__(self, idx):
        cs = self.crop_size_lr or 256
        s = self.scale_factor
        
        # Random crop coords (or 0 if full tile)
        y_lr = random.randint(0, 256 - cs) if cs < 256 else 0
        x_lr = random.randint(0, 256 - cs) if cs < 256 else 0
        
        if self.lr_in_ram:
            # Extract uint16 from cache, then convert to float32
            lr_int = self.lr_cache[idx, :, y_lr:y_lr+cs, x_lr:x_lr+cs]
            lr = (lr_int.float() / 10000.0).clamp(0, 1)
        else:
            lr = self._read_window(self.lr_paths[idx], x_lr, y_lr, cs, cs)

        hr = self._read_window(self.hr_paths[idx], x_lr*s, y_lr*s, cs*s, cs*s)
        
        # Random augmentations (only applied if we're doing crops, skip for full tile usually, but safe to leave)
        if self.crop_size_lr is not None:
            if random.random() > 0.5:
                lr = torch.flip(lr, [-1])
                hr = torch.flip(hr, [-1])
            if random.random() > 0.5:
                lr = torch.flip(lr, [-2])
                hr = torch.flip(hr, [-2])
                
        return lr, hr
