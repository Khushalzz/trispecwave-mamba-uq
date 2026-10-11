"""Diagnostic script to test 1-patch overfitting on TriSpecWave-Mamba-UQ.

Uses Charbonnier and Gradient Edge losses instead of L1.
Disables MTF and SAM losses to isolate the model's ability to learn high frequencies.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("one_patch")

TEACHER_BANDS = [1, 2, 3, 7]  # B2, B3, B4, B8

def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))

def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return 10 * torch.log10((max_val ** 2) / mse)

def run_one_patch_test(config_path="configs/distill_pune.yaml", max_steps=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting 1-patch diagnostic test on {device}...")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 1. Load a single patch directly into VRAM
    cache_dir = Path(cfg.get("paths", {}).get("teacher_cache_dir", "/mnt/pune/teacher_cache"))
    cache_files = sorted(list(cache_dir.glob("*.pt")))
    if not cache_files:
        raise RuntimeError(f"No patches found in {cache_dir}")
    
    pt = torch.load(cache_files[0], map_location="cpu", weights_only=False)
    x = pt["input_patch"].float().unsqueeze(0).to(device)       # [1, 12, 64, 64]
    teacher_hr = pt["teacher_output"].float().unsqueeze(0).to(device)  # [1, 4, 256, 256]

    logger.info(f"Loaded 1 patch into VRAM. Input: {x.shape}, Target: {teacher_hr.shape}")

    # 2. Build Fresh Model (Random Weights)
    model = TriSpecWaveMambaUQ(
        in_channels=cfg.get("in_channels", 12),
        out_channels=cfg.get("out_channels", 12),
        feature_dim=cfg.get("feature_dim", 64),
        spectral_dim=cfg.get("spectral_dim", 64),
        naf_depth=cfg.get("naf_depth", 4),
        spatial_depth=cfg.get("spatial_depth", 2),
        graph_heads=cfg.get("graph_heads", 4),
        dropout=0.0, # Disable dropout for overfitting
        band_wavelengths_nm=cfg.get("band_wavelengths_nm"),
        band_gsd_m=cfg.get("band_gsd_m"),
        upsample_stages=cfg.get("upsample_stages", 2),
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    logger.info("Starting training loop (MTF and SAM disabled, Charbonnier + Gradient active)...")
    start_time = time.time()
    
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        
        student_hr, _ = model(x)
        student_rgbn = student_hr[:, TEACHER_BANDS]

        l_charb = charbonnier_loss(student_rgbn, teacher_hr)
        l_grad = gradient_loss(student_rgbn, teacher_hr)
        
        # Total loss heavily weights edges
        loss = l_charb + 0.5 * l_grad
        
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == max_steps:
            with torch.no_grad():
                current_psnr = psnr(student_rgbn, teacher_hr).item()
            logger.info(
                f"Step [{step:4d}/{max_steps}] | "
                f"Total Loss: {loss.item():.4f} | "
                f"Charbonnier: {l_charb.item():.4f} | "
                f"Gradient: {l_grad.item():.4f} | "
                f"PSNR: {current_psnr:.2f} dB"
            )

    elapsed = time.time() - start_time
    logger.info(f"1-Patch test finished in {elapsed:.1f}s.")
    
    # Generate Visual Comparison
    import matplotlib.pyplot as plt
    
    # 4-band indices for RGB
    RGB = [2, 1, 0] # B4(Red), B3(Green), B2(Blue)
    
    model.eval()
    with torch.no_grad():
        final_student_hr, _ = model(x)
        final_rgbn = final_student_hr[:, TEACHER_BANDS]
    
    bicubic = F.interpolate(x[:, TEACHER_BANDS], size=(256, 256), mode="bicubic").clamp(0, 1)
    
    # Convert to numpy and clip
    lr_rgb = (x[0, RGB].cpu().permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
    bicubic_rgb = (bicubic[0, RGB].cpu().permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
    teacher_rgb = (teacher_hr[0, RGB].cpu().permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
    student_rgb = (final_rgbn[0, RGB].cpu().permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
    residual = np.abs(student_rgb - teacher_rgb).mean(axis=-1)
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(lr_rgb); axes[0].set_title("LR 10m"); axes[0].axis("off")
    axes[1].imshow(bicubic_rgb); axes[1].set_title("Bicubic Baseline"); axes[1].axis("off")
    axes[2].imshow(teacher_rgb); axes[2].set_title("Teacher (Target)"); axes[2].axis("off")
    axes[3].imshow(student_rgb); axes[3].set_title("Student (Charbonnier+Grad)"); axes[3].axis("off")
    axes[4].imshow(residual, cmap="inferno", vmin=0, vmax=0.15); axes[4].set_title("Residual Error"); axes[4].axis("off")
    
    plt.tight_layout()
    outputs_dir = Path("/mnt/pune/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "one_patch_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved visual comparison to {out_path}")
    
    return {"final_psnr": current_psnr}

if __name__ == "__main__":
    run_one_patch_test()
