import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from scripts.sen2neon_dataset import SEN2NEONDataset

def evaluate_sen2neon(checkpoint_path="/mnt/pune/checkpoints/sen2neon/best.pt", samples=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating SEN2NEON model from {checkpoint_path} on {device}...")

    # Load Model
    model = TriSpecWaveMambaUQ().to(device)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict.get("model_state_dict", state_dict.get("model", state_dict)))
    model.eval()
    
    # Load Dataset (Validation split)
    data_dir = "/mnt/pune/sen2neon"
    csv_path = f"{data_dir}/metadata.csv"
    val_ds = SEN2NEONDataset(root_dir=data_dir, csv_path=csv_path, split="val", crop_size_lr=128)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    psnr_scores = []
    
    print(f"Running evaluation over {samples} patches...")
    
    out_dir = Path("/mnt/pune/outputs/stage6")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, (lr, hr) in enumerate(val_loader):
            if i >= samples:
                break
                
            lr = lr.to(device)
            hr = hr.to(device)
            
            with torch.amp.autocast('cuda'):
                out = model(lr)
                pred = out[0] if isinstance(out, tuple) else out
                
            pred = pred.clamp(0, 1)
            
            # Calculate PSNR across all 12 bands
            mse = F.mse_loss(pred, hr)
            psnr = 10 * torch.log10(1.0 / mse)
            psnr_scores.append(psnr.item())
            
            # Save 1 visual comparison (RGB bands: 3, 2, 1)
            if i == 0:
                lr_rgb = lr[0, [3, 2, 1]].cpu().numpy().transpose(1, 2, 0)
                hr_rgb = hr[0, [3, 2, 1]].cpu().numpy().transpose(1, 2, 0)
                pred_rgb = pred[0, [3, 2, 1]].cpu().numpy().transpose(1, 2, 0)
                
                # Scale up LR for visualization
                lr_rgb = np.kron(lr_rgb, np.ones((4, 4, 1)))
                
                # Brighten for visual clarity (S2 reflectance is often dark)
                lr_rgb = np.clip(lr_rgb * 3.0, 0, 1)
                hr_rgb = np.clip(hr_rgb * 3.0, 0, 1)
                pred_rgb = np.clip(pred_rgb * 3.0, 0, 1)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(lr_rgb)
                axes[0].set_title("Input (10m)")
                axes[1].imshow(pred_rgb)
                axes[1].set_title(f"Prediction (2.5m) | PSNR: {psnr.item():.2f}")
                axes[2].imshow(hr_rgb)
                axes[2].set_title("Ground Truth (2.5m)")
                
                for ax in axes:
                    ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(out_dir / "sen2neon_comparison.png", dpi=150)
                plt.close()
                print(f"Saved visual comparison to {out_dir / 'sen2neon_comparison.png'}")
                
    mean_psnr = np.mean(psnr_scores)
    print(f"\n======================================")
    print(f"Stage 6 SEN2NEON Evaluation Results")
    print(f"======================================")
    print(f"Samples Evaluated: {samples}")
    print(f"Average PSNR (12-band): {mean_psnr:.2f} dB")
    print(f"======================================")
    
if __name__ == "__main__":
    evaluate_sen2neon()
