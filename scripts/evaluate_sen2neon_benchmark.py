import os
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from models.rgbn_unet_adapter import RGBNUNetAdapter
import yaml
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def calculate_sam(pred, target):
    B, C, H, W = pred.shape
    p_flat = pred.view(B, C, -1).double()
    t_flat = target.view(B, C, -1).double()
    dot = torch.sum(p_flat * t_flat, dim=1)
    norm_p = torch.linalg.norm(p_flat, dim=1)
    norm_t = torch.linalg.norm(t_flat, dim=1)
    denom = torch.clamp(norm_p * norm_t, min=1e-8)
    cos_theta = torch.clamp(dot / denom, min=-1.0, max=1.0)
    return torch.acos(cos_theta).mean().item()

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
    return lr_tensor, hr_tensor

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting SEN2NEON RGBN Benchmark on {device}")

    # 1. Load Mamba
    with open("configs/sen2neon_6m.yaml") as f:
        cfg = yaml.safe_load(f)
    mamba = TriSpecWaveMambaUQ(
        feature_dim=cfg["feature_dim"], spectral_dim=cfg["spectral_dim"],
        naf_depth=cfg["naf_depth"], spatial_depth=cfg["spatial_depth"],
        band_wavelengths_nm=cfg["band_wavelengths_nm"], band_gsd_m=cfg["band_gsd_m"],
    ).to(device)
    ckpt = torch.load("/mnt/pune/checkpoints/sen2neon_6m_phase1/best.pt", map_location=device, weights_only=True)
    mamba.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()})
    mamba.eval()

    # 2. Load UNet
    unet = RGBNUNetAdapter(in_channels=24, out_channels=4).to(device)
    unet.load_state_dict(torch.load("/mnt/pune/checkpoints/sen2neon_unet_adapter/epoch_3.pt", map_location=device, weights_only=True))
    unet.eval()

    # 3. Load Data
    val_lr, val_hr = load_all_into_ram_safe("/mnt/pune/sen2neon_patches_16x/val", 3632)
    val_loader = DataLoader(TensorDataset(val_lr, val_hr), batch_size=16, shuffle=False)

    # 4. Metrics setup
    rgbn_idx = [1, 2, 3, 7] # Blue, Green, Red, NIR
    
    metrics = {
        "bicubic": {"psnr": PeakSignalNoiseRatio(data_range=1.0).to(device), "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device), "sam": 0.0},
        "mamba": {"psnr": PeakSignalNoiseRatio(data_range=1.0).to(device), "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device), "sam": 0.0},
        "unet": {"psnr": PeakSignalNoiseRatio(data_range=1.0).to(device), "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device), "sam": 0.0},
    }
    batches = 0

    logger.info("Evaluating on 4-Band RGBN (Blue, Green, Red, NIR)...")
    with torch.no_grad():
        for x, target in val_loader:
            x, target = x.to(device), target.to(device)
            target_rgbn = target[:, rgb_idx] if "rgb_idx" in locals() else target[:, rgbn_idx]
            
            # Bicubic
            bicubic = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
            bic_rgbn = bicubic[:, rgbn_idx]
            
            # Mamba Phase 1
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mamba_out = mamba(x)
            mamba_pred = (mamba_out[0] if isinstance(mamba_out, tuple) else mamba_out).clamp(0, 1)
            mamba_rgbn = mamba_pred[:, rgbn_idx]
            
            # UNet Phase 2
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                unet_pred = unet(mamba_pred, bicubic).clamp(0, 1)
                
            # Update Metrics
            metrics["bicubic"]["psnr"].update(bic_rgbn, target_rgbn)
            metrics["bicubic"]["ssim"].update(bic_rgbn, target_rgbn)
            metrics["bicubic"]["sam"] += calculate_sam(bic_rgbn, target_rgbn)
            
            metrics["mamba"]["psnr"].update(mamba_rgbn, target_rgbn)
            metrics["mamba"]["ssim"].update(mamba_rgbn, target_rgbn)
            metrics["mamba"]["sam"] += calculate_sam(mamba_rgbn, target_rgbn)
            
            metrics["unet"]["psnr"].update(unet_pred, target_rgbn)
            metrics["unet"]["ssim"].update(unet_pred, target_rgbn)
            metrics["unet"]["sam"] += calculate_sam(unet_pred, target_rgbn)
            
            batches += 1

    logger.info("=" * 50)
    logger.info("FINAL SEN2NEON RGBN (4-BAND) VALIDATION METRICS (3,632 patches)")
    logger.info("--------------------------------------------------")
    logger.info("1. BICUBIC BASELINE:")
    logger.info(f"   PSNR: {metrics['bicubic']['psnr'].compute().item():.4f}")
    logger.info(f"   SSIM: {metrics['bicubic']['ssim'].compute().item():.4f}")
    logger.info(f"   SAM:  {metrics['bicubic']['sam'] / batches:.4f}")
    logger.info("--------------------------------------------------")
    logger.info("2. 6M MAMBA (PHASE 1):")
    logger.info(f"   PSNR: {metrics['mamba']['psnr'].compute().item():.4f}")
    logger.info(f"   SSIM: {metrics['mamba']['ssim'].compute().item():.4f}")
    logger.info(f"   SAM:  {metrics['mamba']['sam'] / batches:.4f}")
    logger.info("--------------------------------------------------")
    logger.info("3. 6M MAMBA + U-NET DETAILER (PHASE 2):")
    logger.info(f"   PSNR: {metrics['unet']['psnr'].compute().item():.4f}")
    logger.info(f"   SSIM: {metrics['unet']['ssim'].compute().item():.4f}")
    logger.info(f"   SAM:  {metrics['unet']['sam'] / batches:.4f}")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_benchmark()
