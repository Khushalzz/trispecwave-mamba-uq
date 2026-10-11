import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from skimage.exposure import match_histograms
import math

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def compute_psnr_masked(pred, target, mask):
    # pred, target, mask are flat tensors or 1D arrays of valid pixels
    mse = F.mse_loss(pred[mask], target[mask])
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse.item())

def load_all_into_ram_safe(chunk_dir, num_patches):
    logger.info(f"Loading {num_patches} patches from {chunk_dir}...")
    lr_tensor = torch.empty((num_patches, 12, 64, 64), dtype=torch.float32)
    hr_tensor = torch.empty((num_patches, 12, 256, 256), dtype=torch.float32)
    
    idx = 0
    files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.pt')])
    for f in files:
        if idx >= num_patches:
            break
        path = os.path.join(chunk_dir, f)
        try:
            data = torch.load(path, map_location='cpu', weights_only=True)
            lr_chunk = (data["lr"].float() / 10000.0).clamp(0, 1)
            hr_chunk = (data["hr"].float() / 10000.0).clamp(0, 1)
            b = lr_chunk.shape[0]
            lr_tensor[idx:idx+b] = lr_chunk
            hr_tensor[idx:idx+b] = hr_chunk
            idx += b
        except Exception as e:
            logger.warning(f"Error loading {f}: {e}")
    return lr_tensor[:idx], hr_tensor[:idx]

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting Official SEN2NEON Benchmark Evaluation on {device}")
    
    # 1. Load Model
    model = TriSpecWaveMambaUQ(
        in_channels=12,
        out_channels=12,
        feature_dim=128,
        spectral_dim=64,
        naf_depth=8,
        spatial_depth=4
    ).to(device)
    
    ckpt_path = "/mnt/pune/checkpoints/sen2neon_6m_phase1/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('_orig_mod.', '')] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # 2. Load Test Data
    test_dir = "/mnt/pune/sen2neon_patches_16x/val"
    lr_test, hr_test = load_all_into_ram_safe(test_dir, 3632)
    test_loader = DataLoader(TensorDataset(lr_test, hr_test), batch_size=8, shuffle=False)
    
    # Metrics aggregators
    # Official 6-band (20m) with Histogram Matching
    official_mse_sum = 0.0
    official_valid_pixels = 0
    
    # Strict 12-band Raw
    raw_mse_sum = 0.0
    raw_valid_pixels = 0
    
    band_indices_20m = [4, 5, 6, 8, 10, 11]
    
    logger.info("Starting inference loop...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i, (lr, hr) in enumerate(test_loader):
                lr, hr = lr.to(device), hr.to(device)
                
                # Forward pass
                out = model(lr)
                pred = out[0] if isinstance(out, tuple) else out
                pred = pred.clamp(0, 1).float()
                
                # --- Strict 12-Band Raw Score ---
                raw_mask = (hr > 0)
                if raw_mask.any():
                    raw_mse_sum += F.mse_loss(pred[raw_mask], hr[raw_mask], reduction='sum').item()
                    raw_valid_pixels += raw_mask.sum().item()
                
                # --- Official 6-Band (20m) Histogram Matched Score ---
                pred_20m = pred[:, band_indices_20m].cpu().numpy()
                hr_20m = hr[:, band_indices_20m].cpu().numpy()
                mask_20m = (hr_20m > 0)
                
                B, C, H, W = pred_20m.shape
                pred_matched = np.zeros_like(pred_20m)
                
                for b_idx in range(B):
                    for c_idx in range(C):
                        p = pred_20m[b_idx, c_idx]
                        t = hr_20m[b_idx, c_idx]
                        # Match histograms ignoring zeros if possible
                        # Skimage match_histograms applies globally, we can just apply it directly
                        matched = match_histograms(p, t)
                        pred_matched[b_idx, c_idx] = matched
                        
                pred_matched_tensor = torch.from_numpy(pred_matched).to(device)
                hr_20m_tensor = torch.from_numpy(hr_20m).to(device)
                mask_20m_tensor = torch.from_numpy(mask_20m).to(device)
                
                if mask_20m_tensor.any():
                    official_mse_sum += F.mse_loss(pred_matched_tensor[mask_20m_tensor], hr_20m_tensor[mask_20m_tensor], reduction='sum').item()
                    official_valid_pixels += mask_20m_tensor.sum().item()
                    
                if i % 10 == 0:
                    logger.info(f"Processed batch {i}/{len(test_loader)}")
                    
    # Final Calculation
    raw_mse = raw_mse_sum / raw_valid_pixels if raw_valid_pixels > 0 else float('inf')
    raw_psnr = 10 * math.log10(1.0 / raw_mse) if raw_mse > 0 else float('inf')
    
    off_mse = official_mse_sum / official_valid_pixels if official_valid_pixels > 0 else float('inf')
    off_psnr = 10 * math.log10(1.0 / off_mse) if off_mse > 0 else float('inf')
    
    logger.info("="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"1. STRICT 12-BAND RAW PSNR : {raw_psnr:.4f} dB")
    logger.info(f"2. OFFICIAL 6-BAND HM PSNR : {off_psnr:.4f} dB")
    logger.info("="*50)

if __name__ == "__main__":
    run_evaluation()
