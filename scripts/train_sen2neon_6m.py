import os
import sys
import yaml
import time
import logging
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from scripts.sen2neon_ram_dataset import SEN2NEONRamDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def make_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "bias" in name or "norm" in name or "beta" in name or "gamma" in name \
                or "scale" in name or "edge_logits" in name or "wavelengths" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]

def charbonnier(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target)**2 + eps))

def gradient_loss(pred, target):
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

def worker_init_fn(wid):
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["GDAL_NUM_THREADS"] = "1"
    seed = (torch.initial_seed() + wid) % 2**32
    random.seed(seed)
    np.random.seed(seed)

def run_profile_pass(model, device, micro_batch_size):
    """Run a single forward/backward pass and measure memory."""
    logger.info(f"--- Profiling Microbatch Size = {micro_batch_size} ---")
    
    # Enable explicit checkpointing on sequential blocks
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing'):
            m.gradient_checkpointing = True

    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    dummy_x = torch.randn(micro_batch_size, 12, 256, 256, device=device, dtype=torch.float32)
    dummy_t = torch.randn(micro_batch_size, 12, 1024, 1024, device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    start = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(dummy_x)
        pred = out[0] if isinstance(out, tuple) else out
    
    loss = charbonnier(pred.float(), dummy_t)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    
    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    peak_res = torch.cuda.max_memory_reserved() / (1024**3)
    
    logger.info(f"Profile Result: Peak Allocated = {peak_mem:.2f} GB, Peak Reserved = {peak_res:.2f} GB, Time = {elapsed:.3f}s")
    return peak_res

def train_sen2neon_6m(config_path="configs/sen2neon_6m.yaml", profile_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Stage 7 SEN2NEON 6.17M Training on {device}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 1. Build Model from Scratch
    model = TriSpecWaveMambaUQ(
        feature_dim=cfg["feature_dim"],
        spectral_dim=cfg["spectral_dim"],
        naf_depth=cfg["naf_depth"],
        spatial_depth=cfg["spatial_depth"],
        band_wavelengths_nm=cfg["band_wavelengths_nm"],
        band_gsd_m=cfg["band_gsd_m"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built. Total parameters: {total_params:,}")

    # 2. Profile VRAM & Set Checkpointing
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing'):
            m.gradient_checkpointing = True
            
    micro_batch_size = cfg.get("micro_batch_size", 1)
    peak_res = run_profile_pass(model, device, micro_batch_size)
    
    if profile_only:
        logger.info("Profile complete. Exiting due to profile_only=True")
        return

    # Optional safety check
    gpu_props = torch.cuda.get_device_properties(device)
    total_vram = gpu_props.total_memory / (1024**3)
    logger.info(f"GPU: {gpu_props.name}, Total VRAM: {total_vram:.2f} GB")
    if peak_res > total_vram * 0.9:
        logger.error("VRAM usage is critically high! Reduce micro_batch_size.")
        raise RuntimeError("OOM risk detected during profiling.")

    # 3. Setup Dataset & DataLoader
    fuse_dir = "/mnt/pune/sen2neon"
    local_dir = "/tmp/sen2neon"
    if not os.path.exists(local_dir) and os.path.exists(fuse_dir):
        logger.info(f"Copying {fuse_dir} to {local_dir}...")
        os.system(f"cp -r {fuse_dir} /tmp/")
    
    csv_path = f"{local_dir}/metadata.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata not found at {csv_path}")

    train_ds = SEN2NEONRamDataset(local_dir, csv_path, split="train", crop_size_lr=None)
    val_ds = SEN2NEONRamDataset(local_dir, csv_path, split="val", crop_size_lr=None)

    train_loader = DataLoader(
        train_ds, batch_size=micro_batch_size, shuffle=True,
        num_workers=4, prefetch_factor=2, persistent_workers=True,
        pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # 4. Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        make_param_groups(model, cfg["weight_decay"]),
        lr=cfg["learning_rate"],
    )

    max_updates = cfg["max_updates"]
    warmup_updates = cfg["warmup_updates"]
    accum_steps = cfg["accumulation_steps"]
    
    def lr_lambda(current_update):
        if current_update < warmup_updates:
            return float(current_update) / float(max(1, warmup_updates))
        progress = float(current_update - warmup_updates) / float(max(1, max_updates - warmup_updates))
        min_ratio = cfg["min_lr"] / cfg["learning_rate"]
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Training Loop
    logger.info(f"Starting Training: {max_updates} Updates, Accumulation={accum_steps}, Effective Batch={micro_batch_size*accum_steps}")
    
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    
    # We track forward passes (steps) to trigger optimizer updates
    forward_steps = 0
    updates_completed = 0
    
    while updates_completed < max_updates:
        try:
            x, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, target = next(train_iter)

        forward_steps += 1
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out

        loss = (charbonnier(pred.float(), target.float()) + 0.20 * gradient_loss(pred.float(), target.float()))
        loss = loss / accum_steps
        loss.backward()

        if forward_steps % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates_completed += 1

            if updates_completed % cfg["log_every_updates"] == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(f"Update [{updates_completed}/{max_updates}] Loss: {loss.item()*accum_steps:.4f} | LR: {lr:.2e}")

            if updates_completed % cfg["val_every_updates"] == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vx, vt in val_loader:
                        vx, vt = vx.to(device), vt.to(device)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            vout = model(vx)
                            vp = vout[0] if isinstance(vout, tuple) else vout
                        val_loss += charbonnier(vp.float(), vt.float()).item()
                
                val_loss /= len(val_loader)
                logger.info(f"--> Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_dir = "/mnt/pune/checkpoints/sen2neon_6m"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({
                        "update": updates_completed,
                        "model_state_dict": model.state_dict(),
                        "val_loss": best_val_loss
                    }, f"{save_dir}/best.pt")
                    logger.info(f"--> Saved best checkpoint to {save_dir}/best.pt")
                model.train()

    logger.info("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    train_sen2neon_6m(profile_only=args.profile_only)
