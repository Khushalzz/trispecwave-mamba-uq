import os
import yaml
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from scripts.sen2neon_dataset import SEN2NEONDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))

def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

def train_stage6_sen2neon(config_path="configs/distill_pune.yaml", steps=10000, lr=1e-4, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Stage 6 SEN2NEON Training on {device}...")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 1. Load Backbone with Warm Start
    model = TriSpecWaveMambaUQ(
        in_channels=cfg.get("in_channels", 12),
        out_channels=cfg.get("out_channels", 12),
        feature_dim=cfg.get("feature_dim", 64),
        spectral_dim=cfg.get("spectral_dim", 64),
        naf_depth=cfg.get("naf_depth", 4),
        spatial_depth=cfg.get("spatial_depth", 2),
        graph_heads=cfg.get("graph_heads", 4),
        band_wavelengths_nm=cfg.get("band_wavelengths_nm"),
        band_gsd_m=cfg.get("band_gsd_m"),
    ).to(device)
    
    ckpt_path = "/mnt/pune/checkpoints/edge_finetune/best.pt"
    if not Path(ckpt_path).exists():
        logger.warning(f"Missing backbone checkpoint at {ckpt_path}. Training from scratch!")
    else:
        logger.info(f"Loading warm start from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
    
    model.train()
    
    # 2. Build Datasets
    fuse_data_dir = "/mnt/pune/sen2neon"
    local_data_dir = "/tmp/sen2neon"
    csv_path = f"{local_data_dir}/metadata.csv"
    
    # Wait until dataset download finishes
    if not os.path.exists(f"{fuse_data_dir}/metadata.csv"):
        raise FileNotFoundError(f"Missing SEN2NEON metadata at {fuse_data_dir}. Did the download finish?")
        
    logger.info("Copying dataset from slow FUSE volume to extremely fast local NVMe (/tmp)...")
    os.system(f"cp -r {fuse_data_dir} /tmp/")
    logger.info("Copy complete! This will eliminate the IO bottleneck.")
        
    train_ds = SEN2NEONDataset(root_dir=local_data_dir, csv_path=csv_path, split="train", crop_size_lr=64)
    val_ds = SEN2NEONDataset(root_dir=local_data_dir, csv_path=csv_path, split="val", crop_size_lr=128)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size//4 or 1, shuffle=False, num_workers=4)
    
    train_iter = iter(train_loader)
    
    # 3. Fresh Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.cuda.amp.GradScaler()
    
    logger.info(f"Training for {steps} steps...")
    start_time = time.time()
    best_val_loss = float("inf")
    
    for step in range(1, steps + 1):
        try:
            x, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, target = next(train_iter)
            
        x = x.to(device)
        target = target.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            # TriSpecWaveMambaUQ outputs tuple of (mean, logvar) if not return_aux
            out = model(x)
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out
                
            l_charb = charbonnier_loss(pred, target)
            l_grad = gradient_loss(pred, target)
            
            # Loss equally weights structure and fidelity across ALL 12 bands
            loss = l_charb + 0.20 * l_grad
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if step % 100 == 0 or step == steps:
            logger.info(f"Step [{step:4d}/{steps}] Loss: {loss.item():.4f} (Charb: {l_charb.item():.4f}, Grad: {l_grad.item():.4f})")
            
        if step % 1000 == 0 or step == steps:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_x, v_target in val_loader:
                    v_x = v_x.to(device)
                    v_target = v_target.to(device)
                    
                    with torch.cuda.amp.autocast():
                        v_out = model(v_x)
                        v_pred = v_out[0] if isinstance(v_out, tuple) else v_out
                        v_l = charbonnier_loss(v_pred, v_target) + 0.20 * gradient_loss(v_pred, v_target)
                        val_loss += v_l.item()
                        
            val_loss /= len(val_loader)
            logger.info(f"--> Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dir = "/mnt/pune/checkpoints/sen2neon"
                os.makedirs(save_dir, exist_ok=True)
                save_path = f"{save_dir}/best.pt"
                
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }, save_path)
                logger.info(f"--> Saved best checkpoint to {save_path}")
                
            model.train()
            
    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed/3600:.2f} hours.")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()
    train_stage6_sen2neon(steps=args.steps)
