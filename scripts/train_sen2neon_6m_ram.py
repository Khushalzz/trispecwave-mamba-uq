import os
import yaml
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ

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

def load_all_into_ram_safe(chunk_dir, num_patches):
    logger.info(f"Pre-allocating {num_patches} patches in Float32 RAM...")
    lr_tensor = torch.empty((num_patches, 12, 64, 64), dtype=torch.float32)
    hr_tensor = torch.empty((num_patches, 12, 256, 256), dtype=torch.float32)
    
    idx = 0
    files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.pt')])
    for f in files:
        path = os.path.join(chunk_dir, f)
        data = torch.load(path, weights_only=True)
        lr_chunk = (data["lr"].float() / 10000.0).clamp(0, 1)
        hr_chunk = (data["hr"].float() / 10000.0).clamp(0, 1)
        n = len(lr_chunk)
        
        lr_tensor[idx:idx+n] = lr_chunk
        hr_tensor[idx:idx+n] = hr_chunk
        idx += n
        
        # Free memory immediately
        del data, lr_chunk, hr_chunk
        logger.info(f"Loaded {f} into RAM ({idx}/{num_patches})")
        
    logger.info("Pinning memory for zero-copy DMA to GPU...")
    return lr_tensor.pin_memory(), hr_tensor.pin_memory()


def train_ram(config_path="configs/sen2neon_6m.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting ULTRA-FAST RAM Training on {device}")

    with open(config_path) as f: cfg = yaml.safe_load(f)

    model = TriSpecWaveMambaUQ(
        feature_dim=cfg["feature_dim"], spectral_dim=cfg["spectral_dim"],
        naf_depth=cfg["naf_depth"], spatial_depth=cfg["spatial_depth"],
        band_wavelengths_nm=cfg["band_wavelengths_nm"], band_gsd_m=cfg["band_gsd_m"],
    ).to(device)
    
    # --- RESUME PROGRESS FROM INTERRUPTED RUN ---
    resume_ckpt = "/mnt/pune/checkpoints/sen2neon_6m_patches/latest.pt"
    if os.path.exists(resume_ckpt):
        logger.info(f"Resuming weights from {resume_ckpt} (Epoch 2)...")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=True)
        # Strip torch.compile '_orig_mod.' prefix if present
        clean_state_dict = {}
        for k, v in ckpt["model_state_dict"].items():
            clean_k = k.replace("_orig_mod.", "")
            clean_state_dict[clean_k] = v
        model.load_state_dict(clean_state_dict)
    
    # Disable manual checkpointing and use torch.compile
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing'): m.gradient_checkpointing = False
    
    logger.info("Compiling model for maximum speed...")
    model = torch.compile(model, mode="default")

    # Load everything entirely into RAM!
    # Train: 32,672 patches. Val: 3,632 patches.
    fuse_dir = "/mnt/pune/sen2neon_patches_16x"
    
    logger.info("Loading 108GB Train Dataset into RAM...")
    train_lr, train_hr = load_all_into_ram_safe(f"{fuse_dir}/train", 32672)
    
    logger.info("Loading 12GB Val Dataset into RAM...")
    val_lr, val_hr = load_all_into_ram_safe(f"{fuse_dir}/val", 3632)

    train_ds = TensorDataset(train_lr, train_hr)
    val_ds = TensorDataset(val_lr, val_hr)

    BATCH_SIZE = 64
    
    # num_workers=0 because data is already in RAM! Zero IPC overhead.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # EXACT SAME BATCH SIZE to prevent torch.compile recompilation!
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(make_param_groups(model, cfg["weight_decay"]), lr=cfg["learning_rate"])
    
    EPOCHS = 15
    # 32,672 train patches / 64 = ~510 batches per epoch
    total_steps = 510 * EPOCHS
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
        epoch_start = time.time()
        
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

        logger.info(f"Epoch {epoch} Train Time: {time.time() - epoch_start:.2f} seconds")

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
        
        save_dir = "/mnt/pune/checkpoints/sen2neon_6m_ram"
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
    train_ram()
