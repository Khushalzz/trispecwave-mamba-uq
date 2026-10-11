import os
import yaml
import time
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from scripts.sen2neon_patch_dataset import ChunkedPatchDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def make_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim <= 1 or "bias" in name or "norm" in name or "beta" in name or "gamma" in name \
                or "scale" in name or "edge_logits" in name or "wavelengths" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

def charbonnier(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target)**2 + eps))

def gradient_loss(pred, target):
    dx_p, dy_p = pred[:, :, :, 1:] - pred[:, :, :, :-1], pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_t, dy_t = target[:, :, :, 1:] - target[:, :, :, :-1], target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

def worker_init_fn(wid):
    seed = (torch.initial_seed() + wid) % 2**32
    random.seed(seed); np.random.seed(seed)

def train_patches(config_path="configs/sen2neon_6m.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting High-Batch Patch Training on {device}")

    with open(config_path) as f: cfg = yaml.safe_load(f)

    model = TriSpecWaveMambaUQ(
        feature_dim=cfg["feature_dim"], spectral_dim=cfg["spectral_dim"],
        naf_depth=cfg["naf_depth"], spatial_depth=cfg["spatial_depth"],
        band_wavelengths_nm=cfg["band_wavelengths_nm"], band_gsd_m=cfg["band_gsd_m"],
    ).to(device)

    # Disable manual checkpointing and use torch.compile for maximum speed
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing'): m.gradient_checkpointing = False
    
    logger.info("Compiling model for maximum speed...")
    model = torch.compile(model, mode="default")

    # Dataset paths
    fuse_dir = "/mnt/pune/sen2neon_patches_16x"
    local_dir = "/tmp/sen2neon_patches_16x"
    if not os.path.exists(local_dir):
        logger.info(f"Copying extracted chunks to NVMe: {fuse_dir} -> {local_dir}")
        os.system(f"cp -r {fuse_dir} /tmp/")
        
    train_ds = ChunkedPatchDataset(f"{local_dir}/train", shuffle=True, augment=True)
    val_ds = ChunkedPatchDataset(f"{local_dir}/val", shuffle=False, augment=False)

    BATCH_SIZE = 128  # Massive batch size for 256x256 patches on H100
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=4, 
        pin_memory=True, worker_init_fn=worker_init_fn, prefetch_factor=2
    )
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)

    optimizer = torch.optim.AdamW(make_param_groups(model, cfg["weight_decay"]), lr=cfg["learning_rate"])
    
    EPOCHS = 15
    # 32,672 train patches / 128 = ~255 batches per epoch
    total_steps = 255 * EPOCHS
    warmup_steps = 150
    
    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cfg["min_lr"]/cfg["learning_rate"] + 0.5 * (1 - cfg["min_lr"]/cfg["learning_rate"]) * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_loss = float("inf")
    
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        logger.info(f"--- Epoch {epoch}/{EPOCHS} ---")
        
        for step, (x, target) in enumerate(train_loader):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                pred = out[0] if isinstance(out, tuple) else out

            loss = charbonnier(pred.float(), target.float()) + 0.20 * gradient_loss(pred.float(), target.float())
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 50 == 0:
                logger.info(f"Epoch [{epoch}] Step [{step}] Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Validation at end of epoch
        model.eval()
        val_loss = 0.0
        batches = 0
        with torch.no_grad():
            for vx, vt in val_loader:
                vx, vt = vx.to(device, non_blocking=True), vt.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    vout = model(vx)
                    vp = vout[0] if isinstance(vout, tuple) else vout
                val_loss += charbonnier(vp.float(), vt.float()).item()
                batches += 1
                
        val_loss /= max(1, batches)
        logger.info(f"--> Epoch {epoch} Val Loss: {val_loss:.4f}")
        
        save_dir = "/mnt/pune/checkpoints/sen2neon_6m_patches"
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "epoch": epoch, "step": global_step, "model_state_dict": model.state_dict(), "val_loss": val_loss
        }, f"{save_dir}/latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch, "step": global_step, "model_state_dict": model.state_dict(), "val_loss": val_loss
            }, f"{save_dir}/best.pt")
            logger.info(f"--> Saved best checkpoint to {save_dir}/best.pt (val_loss={best_val_loss:.4f})")

if __name__ == "__main__":
    train_patches()
