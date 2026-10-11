import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
import yaml

def generate_good_visual():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    with open("configs/sen2neon_6m.yaml") as f:
        cfg = yaml.safe_load(f)

    model = TriSpecWaveMambaUQ(
        feature_dim=cfg["feature_dim"], spectral_dim=cfg["spectral_dim"],
        naf_depth=cfg["naf_depth"], spatial_depth=cfg["spatial_depth"],
        band_wavelengths_nm=cfg["band_wavelengths_nm"], band_gsd_m=cfg["band_gsd_m"],
    ).to(device)

    ckpt_path = "/mnt/pune/checkpoints/sen2neon_6m_ram/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    clean_state_dict = {}
    for k, v in ckpt["model_state_dict"].items():
        clean_k = k.replace("_orig_mod.", "")
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 2. Load a different chunk
    chunk_path = "/mnt/pune/sen2neon_patches_16x/val/chunk_0002.pt"
    data = torch.load(chunk_path, weights_only=True)
    lr = (data["lr"].float() / 10000.0).clamp(0, 1)
    hr = (data["hr"].float() / 10000.0).clamp(0, 1)

    # 3. Find a highly-textured patch
    best_idx = 0
    max_std = 0
    # Search a different range to guarantee a new image
    for i in range(200, 500):
        std = hr[i, [3, 2, 1]].std().item()
        if std > max_std:
            max_std = std
            best_idx = i

    print(f"Selected patch {best_idx} with std {max_std}")

    x = lr[best_idx:best_idx+1].to(device)
    target = hr[best_idx:best_idx+1].to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        pred = out[0].clamp(0, 1)

    # 4. Generate Visual
    rgb_bands = [3, 2, 1]
    
    lr_img = x[0, rgb_bands].cpu().numpy().transpose(1, 2, 0)
    pred_img = pred[0, rgb_bands].cpu().numpy().transpose(1, 2, 0)
    hr_img = target[0, rgb_bands].cpu().numpy().transpose(1, 2, 0)
    
    p98 = np.percentile(hr_img, 98)
    p98 = max(p98, 1e-4) # SAFEGUARD
    
    lr_vis = np.clip(lr_img / p98, 0, 1)
    pred_vis = np.clip(pred_img / p98, 0, 1)
    hr_vis = np.clip(hr_img / p98, 0, 1)
    
    import torch.nn.functional as F
    lr_bicubic = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)[0, rgb_bands]
    lr_bicubic_vis = np.clip(lr_bicubic.cpu().numpy().transpose(1, 2, 0) / p98, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr_bicubic_vis)
    axes[0].set_title("Bicubic Low-Resolution (256x256)")
    axes[0].axis('off')
    
    axes[1].imshow(pred_vis)
    axes[1].set_title("TriSpecWaveMambaUQ 6.17M")
    axes[1].axis('off')
    
    axes[2].imshow(hr_vis)
    axes[2].set_title("Ground Truth HR (AVIRIS-NG)")
    axes[2].axis('off')
    
    plt.tight_layout()
    out_path = "/mnt/pune/sen2neon_good_visual_2.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print("Successfully generated good visual at", out_path)

if __name__ == "__main__":
    generate_good_visual()
