import os
import math
import logging
import pandas as pd
import numpy as np
import rasterio as rio
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000  # Number of patches per .pt file
GRID_SIZE = 4      # 4x4 grid = 16 patches per full tile
LR_SIZE = 64       # 256 / 4 = 64
HR_SIZE = 256      # 1024 / 4 = 256

def process_tile(row_idx, lr_path, hr_path):
    """Reads a full tile pair and slices it into 16 patches. Runs in worker processes."""
    try:
        with rio.open(lr_path) as ds:
            lr_arr = ds.read()
            lr_nd = ds.nodata
        with rio.open(hr_path) as ds:
            hr_arr = ds.read()
            hr_nd = ds.nodata
            
        if lr_nd is not None:
            lr_arr = np.where(lr_arr == lr_nd, 0, lr_arr)
        if hr_nd is not None:
            hr_arr = np.where(hr_arr == hr_nd, 0, hr_arr)
            
        lr_arr = lr_arr.astype(np.int16)
        hr_arr = hr_arr.astype(np.int16)
        
        lr_patches = []
        hr_patches = []
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                y_lr, x_lr = i * LR_SIZE, j * LR_SIZE
                y_hr, x_hr = i * HR_SIZE, j * HR_SIZE
                
                lr_crop = lr_arr[:, y_lr:y_lr+LR_SIZE, x_lr:x_lr+LR_SIZE]
                hr_crop = hr_arr[:, y_hr:y_hr+HR_SIZE, x_hr:x_hr+HR_SIZE]
                
                # Check for completely blank/zero patches (optional, we keep them for now)
                lr_patches.append(torch.from_numpy(lr_crop))
                hr_patches.append(torch.from_numpy(hr_crop))
                
        return lr_patches, hr_patches
    except Exception as e:
        logger.error(f"Error processing {lr_path}: {e}")
        return [], []

def extract_patches():
    root_dir = Path("/mnt/pune/sen2neon")
    out_dir = Path("/mnt/pune/sen2neon_patches_16x")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = root_dir / "metadata.csv"
    df = pd.read_csv(csv_path)
    
    lr_col = next((c for c in df.columns if "lr" in c.lower()), None)
    hr_col = next((c for c in df.columns if "hr" in c.lower() and "2_5" in c.lower()), None) \
           or next((c for c in df.columns if "hr" in c.lower()), None)
    
    # Shuffle and split 90/10
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.9)
    splits = {
        "train": df.iloc[:split_idx],
        "val": df.iloc[split_idx:]
    }
    
    for split_name, split_df in splits.items():
        logger.info(f"Extracting {split_name} split ({len(split_df)} tiles)...")
        (out_dir / split_name).mkdir(exist_ok=True)
        
        lr_paths = [str(root_dir / str(p).strip("/")) for p in split_df[lr_col]]
        hr_paths = [str(root_dir / str(p).strip("/")) for p in split_df[hr_col]]
        
        chunk_idx = 0
        current_lr_chunk = []
        current_hr_chunk = []
        
        def save_chunk():
            nonlocal chunk_idx, current_lr_chunk, current_hr_chunk
            if not current_lr_chunk: return
            
            lr_tensor = torch.stack(current_lr_chunk)
            hr_tensor = torch.stack(current_hr_chunk)
            
            save_path = out_dir / split_name / f"chunk_{chunk_idx:04d}.pt"
            torch.save({"lr": lr_tensor, "hr": hr_tensor}, save_path)
            
            logger.info(f"Saved {save_path} ({len(lr_tensor)} patches)")
            chunk_idx += 1
            current_lr_chunk = []
            current_hr_chunk = []

        # Process in parallel using threads (rasterio releases GIL, avoids IPC deadlocks)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_tile, i, lr_paths[i], hr_paths[i]) for i in range(len(split_df))]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name} tiles"):
                lr_p, hr_p = future.result()
                current_lr_chunk.extend(lr_p)
                current_hr_chunk.extend(hr_p)
                
                while len(current_lr_chunk) >= CHUNK_SIZE:
                    # Keep exactly CHUNK_SIZE, roll the rest over
                    excess_lr = current_lr_chunk[CHUNK_SIZE:]
                    excess_hr = current_hr_chunk[CHUNK_SIZE:]
                    
                    current_lr_chunk = current_lr_chunk[:CHUNK_SIZE]
                    current_hr_chunk = current_hr_chunk[:CHUNK_SIZE]
                    save_chunk()
                    
                    current_lr_chunk = excess_lr
                    current_hr_chunk = excess_hr
                    
        # Save remainder
        if current_lr_chunk:
            save_chunk()
            
    logger.info("Extraction completely finished!")

if __name__ == "__main__":
    extract_patches()
