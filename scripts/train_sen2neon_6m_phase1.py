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

def make_param_groups_differential(model, base_lr, tail_lr, weight_decay):
    """
    Applies differential learning rates.
    The feature backbone gets a very small learning rate (to protect features).
    The upsampler and tail get a higher learning rate (to rapidly learn to draw sharp edges).
    """
    backbone_decay, backbone_no_decay = [], []
    tail_decay, tail_no_decay = [], []
    
    # Identify tail components by name (adjust if your architecture names differ slightly)
    tail_keywords = ['upsampler', 'tail', 'reconstruction', 'skip', 'head', 'fusion']
    
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        
        is_tail = any(k in name for k in tail_keywords)
        is_no_decay = p.ndim <= 1 or "bias" in name or "norm" in name or "beta" in name or "gamma" in name \
                      or "scale" in name or "edge_logits" in name or "wavelengths" in name

        if is_tail:
            if is_no_decay: tail_no_decay.append(p)
            else: tail_decay.append(p)
        else:
            if is_no_decay: backbone_no_decay.append(p)
            else: backbone_decay.append(p)

    return [
        {"params": backbone_decay, "weight_decay": weight_decay, "lr": base_lr},
        {"params": backbone_no_decay, "weight_decay": 0.0, "lr": base_lr},
        {"params": tail_decay, "weight_decay": weight_decay, "lr": tail_lr},
        {"params": tail_no_decay, "weight_decay": 0.0, "lr": tail_lr}
    ]

def shifted_window_total_loss(pred, hr, mask, max_shift=1):
    """
    Phase 1 Core Fix:
    Forgives 1-pixel camera jitter by computing L1 + Gradient Loss across 
    shifted versions of the target, and taking the minimum loss per-pixel.
    Also strictly masks out 'nodata' black borders so gradients don't explode.
    """
    losses = []
    
    # Pad for gradient calculation to keep spatial dims same (B, C, H, W)
    pred_pad = F.pad(pred, (0, 1, 0, 1), mode='replicate')
    dx_p = pred_pad[:, :, :-1, 1:] - pred_pad[:, :, :-1, :-1] # Shape: (B, C, H, W)
    dy_p = pred_pad[:, :, 1:, :-1] - pred_pad[:, :, :-1, :-1] # Shape: (B, C, H, W)

    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            hr_shifted = torch.roll(hr, shifts=(dx, dy), dims=(2, 3))
            mask_shifted = torch.roll(mask, shifts=(dx, dy), dims=(2, 3))
            
            valid_pixels = mask * mask_shifted
            
            # 1. Base L1 (Replacing Charbonnier to be sharper)
            l1 = F.l1_loss(pred, hr_shifted, reduction='none')
            
            # 2. Gradient / High-Freq Loss
            hr_pad = F.pad(hr_shifted, (0, 1, 0, 1), mode='replicate')
            dx_t = hr_pad[:, :, :-1, 1:] - hr_pad[:, :, :-1, :-1]
            dy_t = hr_pad[:, :, 1:, :-1] - hr_pad[:, :, :-1, :-1]
            
            grad_loss = F.l1_loss(dx_p, dx_t, reduction='none') + F.l1_loss(dy_p, dy_t, reduction='none')
            
            # Combine (Heavily weight gradients to force sharpness)
            total_pixel_loss = (l1 + 0.5 * grad_loss) * valid_pixels
            losses.append(total_pixel_loss)
            
    # Best-of-K spatial: Take the MINIMUM loss across all shifts per pixel
    min_loss, _ = torch.stack(losses, dim=0).min(dim=0)
    
    # Average over valid pixels
    return min_loss.sum() / (mask.sum() + 1e-8)

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
        
        del data, lr_chunk, hr_chunk
        logger.info(f"Loaded {f} into RAM ({idx}/{num_patches})")
        
    logger.info("Pinning memory for zero-copy DMA to GPU...")
    return lr_tensor.pin_memory(), hr_tensor.pin_memory()

def train_phase1(config_path="configs/sen2neon_6m.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Phase 1 (Shifted L1 + Nodata Mask) Training on {device}")

    with open(config_path) as f: cfg = yaml.safe_load(f)

    model = TriSpecWaveMambaUQ(
        feature_dim=cfg["feature_dim"], spectral_dim=cfg["spectral_dim"],
        naf_depth=cfg["naf_depth"], spatial_depth=cfg["spatial_depth"],
        band_wavelengths_nm=cfg["band_wavelengths_nm"], band_gsd_m=cfg["band_gsd_m"],
    ).to(device)
    
    # --- LOAD THE BLURRY BEST.PT AS OUR STARTING POINT ---
    resume_ckpt = "/mnt/pune/checkpoints/sen2neon_6m_ram/best.pt"
    if os.path.exists(resume_ckpt):
        logger.info(f"Loading converged weights from {resume_ckpt}...")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=True)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(clean_state_dict)
    else:
        logger.warning(f"Could not find {resume_ckpt}! Training from scratch.")
    
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing'): m.gradient_checkpointing = False
    
    logger.info("Compiling model for maximum speed...")
    model = torch.compile(model, mode="default")

    fuse_dir = "/mnt/pune/sen2neon_patches_16x"
    
    logger.info("Loading Train Dataset into RAM...")
    train_lr, train_hr = load_all_into_ram_safe(f"{fuse_dir}/train", 32672)
    
    logger.info("Loading Val Dataset into RAM...")
    val_lr, val_hr = load_all_into_ram_safe(f"{fuse_dir}/val", 3632)

    train_ds = TensorDataset(train_lr, train_hr)
    val_ds = TensorDataset(val_lr, val_hr)

    BATCH_SIZE = 64
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- PHASE 1 DIFFERENTIAL LEARNING RATES ---
    # Give the backbone a tiny LR so it doesn't forget features
    # Give the tail a strong LR so it aggressively learns sharp edges
    base_lr = 1e-5
    tail_lr = 1e-4
    
    optimizer = torch.optim.AdamW(
        make_param_groups_differential(model, base_lr, tail_lr, cfg["weight_decay"])
    )
    
    # --- EXACTLY 5 EPOCHS AS REQUESTED ---
    EPOCHS = 5
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = 100 # Short warmup
    
    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * (0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_loss = float("inf")
    
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        logger.info(f"--- Epoch {epoch}/{EPOCHS} ---")
        epoch_start = time.time()
        
        for step, (x, target) in enumerate(train_loader):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # --- PHASE 1 DYNAMIC NODATA MASK ---
            # If a pixel in the target is exactly 0 across all bands, it's nodata (black border)
            valid_mask = (target > 0).any(dim=1, keepdim=True).float() 

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                pred = out[0] if isinstance(out, tuple) else out
            
            # Use the new Shifted Window Total Loss
            loss = shifted_window_total_loss(pred.float(), target.float(), valid_mask, max_shift=1)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 50 == 0:
                logger.info(f"Epoch [{epoch}] Step [{step}] Loss: {loss.item():.4f} | Tail LR: {scheduler.get_last_lr()[2]:.2e}")

        logger.info(f"Epoch {epoch} Train Time: {time.time() - epoch_start:.2f} seconds")

        # Validation
        model.eval()
        val_loss = 0.0
        batches = 0
        with torch.no_grad():
            for vx, vt in val_loader:
                vx, vt = vx.to(device, non_blocking=True), vt.to(device, non_blocking=True)
                v_mask = (vt > 0).any(dim=1, keepdim=True).float()
                
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    vout = model(vx)
                    vp = vout[0] if isinstance(vout, tuple) else vout
                    
                val_loss += shifted_window_total_loss(vp.float(), vt.float(), v_mask, max_shift=1).item()
                batches += 1
                
        val_loss /= max(1, batches)
        logger.info(f"--> Epoch {epoch} Val Loss: {val_loss:.4f}")
        
        save_dir = "/mnt/pune/checkpoints/sen2neon_6m_phase1"
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
    train_phase1()
