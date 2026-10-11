import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from models.rgbn_unet_adapter import RGBNUNetAdapter

def generate_pune_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Pune evaluation on {device}")
    
    # 1. Load Pune Data from teacher cache
    cache_dir = "/mnt/pune/teacher_cache"
    if not os.path.exists(cache_dir):
        print("Pune data not found in teacher_cache!")
        return
        
    import glob
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not files:
        print("No files found!")
        return
        
    # Pick a random patch (e.g. index 42)
    patch_file = files[min(42, len(files)-1)]
    data = torch.load(patch_file, map_location="cpu", weights_only=True)
    
    # teacher_cache format: {"input_patch": (12, 64, 64)}
    lr_patch = data["input_patch"].unsqueeze(0).float().to(device)
    
    # 2. Load 6.17M Mamba Student (Phase 1)
    print("Loading Phase 1 6.17M Mamba...")
    with open("configs/sen2neon_6m.yaml") as f:
        cfg = yaml.safe_load(f)

    mamba = TriSpecWaveMambaUQ(
        feature_dim=cfg["feature_dim"], spectral_dim=cfg["spectral_dim"],
        naf_depth=cfg["naf_depth"], spatial_depth=cfg["spatial_depth"],
        band_wavelengths_nm=cfg["band_wavelengths_nm"], band_gsd_m=cfg["band_gsd_m"],
    ).to(device)

    ckpt_path = "/mnt/pune/checkpoints/sen2neon_6m_phase1/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    clean_state_dict = {}
    for k, v in ckpt["model_state_dict"].items():
        clean_state_dict[k.replace("_orig_mod.", "")] = v
    mamba.load_state_dict(clean_state_dict)
    mamba.eval()
    
    # 3. Load UNet Adapter
    print("Loading U-Net Detailer...")
    unet = RGBNUNetAdapter(in_channels=24, out_channels=4).to(device)
    unet.load_state_dict(torch.load("/mnt/pune/checkpoints/sen2neon_unet_adapter/epoch_3.pt", map_location=device, weights_only=True))
    unet.eval()
    
    rgb_bands = [3, 2, 1]

    # Inference
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            mamba_out = mamba(lr_patch)
        mamba_pred = mamba_out[0] if isinstance(mamba_out, tuple) else mamba_out
        mamba_pred = mamba_pred.clamp(0, 1)
        mamba_rgb = mamba_pred[0, rgb_bands].cpu().numpy().transpose(1, 2, 0)
        
        # Bicubic
        lr_bicubic = F.interpolate(lr_patch, scale_factor=4, mode='bicubic', align_corners=False)
        bicubic_rgb = lr_bicubic[0, rgb_bands].cpu().numpy().transpose(1, 2, 0)
        
        # UNet
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            unet_out = unet(mamba_pred, lr_bicubic).clamp(0, 1)
            
        # UNet outputs 4 channels (RGBN). RGB indices are 2, 1, 0 (Red, Green, Blue)
        unet_rgb = unet_out[0, [2, 1, 0]].cpu().float().numpy().transpose(1, 2, 0)
        
    # 5. Visualize
    print("Generating visual...")
    
    # Normalization for visualization (using 98th percentile of Bicubic as reference)
    p2 = np.percentile(bicubic_rgb, 2)
    p98 = np.percentile(bicubic_rgb, 98)
    norm = lambda x: np.clip((x - p2) / max(p98 - p2, 1e-4), 0, 1)
    
    bic_vis = norm(bicubic_rgb)
    mamba_vis = norm(mamba_rgb)
    unet_vis = norm(unet_rgb)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(bic_vis)
    axes[0].set_title("Bicubic Low-Res (Pune, India)")
    axes[0].axis('off')
    
    axes[1].imshow(mamba_vis)
    axes[1].set_title("TriSpecWaveMambaUQ (6M Base)")
    axes[1].axis('off')
    
    axes[2].imshow(unet_vis)
    axes[2].set_title("6M Base + U-Net Detailer")
    axes[2].axis('off')
    
    plt.tight_layout()
    out_path = "/mnt/pune/pune_unet_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print("Successfully generated Pune comparison at", out_path)


if __name__ == "__main__":
    generate_pune_comparison()
