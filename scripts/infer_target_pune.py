import os
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
import numpy as np
import matplotlib.pyplot as plt
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

def infer_target():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_dir = "/mnt/pune/raw/S2A_MSIL2A_20240304T052721_R105_T43QCA_20240304T101311"
    lat, lon = 18.4388, 73.7728
    patch_size = 128
    
    # 1. We must read the 10m bands directly to define the window
    b2_file = f"{scene_dir}/B2.tif"
    with rasterio.open(b2_file) as src:
        xs, ys = transform("epsg:4326", src.crs, [lon], [lat])
        utm_x, utm_y = xs[0], ys[0]
        row, col = src.index(utm_x, utm_y)
        
        start_row = max(0, row - patch_size // 2)
        start_col = max(0, col - patch_size // 2)
        window = Window(start_col, start_row, patch_size, patch_size)
        window_bounds = src.window_bounds(window)
        
    bands = []
    for b in BAND_NAMES:
        f = f"{scene_dir}/{b}.tif"
        with rasterio.open(f) as src:
            # For non-10m bands, we use out_shape to resample them to the 10m patch size (128x128)
            # We also read using the bounding box of the window
            win = rasterio.windows.from_bounds(*window_bounds, transform=src.transform)
            
            data = src.read(
                1, 
                window=win,
                out_shape=(patch_size, patch_size),
                resampling=rasterio.enums.Resampling.bilinear
            )
            bands.append(data)
            
    # Stack into [12, H, W]
    img = np.stack(bands, axis=0).astype(np.float32)
    # Scale from [0, 10000] to [0, 1]
    img = img / 10000.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    # 2. Load the 6M model
    model = TriSpecWaveMambaUQ(
        in_channels=12,
        out_channels=12,
        feature_dim=128,
        spectral_dim=64,
        naf_depth=8,
        spatial_depth=4
    ).to(device)
    
    ckpt_path = "/mnt/pune/checkpoints/sen2neon_6m_phase1/best.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(new_state_dict)
        
    model.eval()
    
    # 3. Load SRGAN
    try:
        from opensr_srgan import load_inference_model
        srgan = load_inference_model("RGB-NIR").to(device).eval()
    except Exception as e:
        print("SRGAN not found:", e)
        srgan = None
        
    # 4. Infer
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(img_tensor)
        pred = out[0] if isinstance(out, tuple) else out
        pred = pred.clamp(0, 1)
        
        bicubic = F.interpolate(img_tensor, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        
        if srgan:
            # SRGAN needs B2, B3, B4, B8 -> Indices 1, 2, 3, 7
            srgan_in = img_tensor[:, [1, 2, 3, 7], :, :].float() * 10000.0
            srgan_out = srgan(srgan_in).clamp(0, 10000) / 10000.0
        else:
            srgan_out = torch.zeros((1, 4, patch_size*4, patch_size*4), device=device)
            
    # RGB indices: B4=3, B3=2, B2=1 (in 0-indexed BAND_NAMES array)
    rgb_idx = [3, 2, 1]
    
    # Normalize for display
    # Use 98th percentile of bicubic for stable normalization
    bic_rgb = bicubic[0, rgb_idx].cpu().numpy().transpose(1, 2, 0)
    p2, p98 = np.percentile(bic_rgb, 2), np.percentile(bic_rgb, 98)
    norm = lambda x: np.clip((x - p2) / max(p98 - p2, 1e-4), 0, 1)
    
    pred_rgb = pred[0, rgb_idx].cpu().float().numpy().transpose(1, 2, 0)
    # SRGAN output is B2, B3, B4, B8 (idx 0,1,2,3). RGB is B4,B3,B2 -> 2, 1, 0
    srgan_rgb = srgan_out[0, [2, 1, 0]].cpu().float().numpy().transpose(1, 2, 0)
    
    lr_rgb = img_tensor[0, rgb_idx].cpu().numpy().transpose(1, 2, 0)
    
    # Plot 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(norm(F.interpolate(img_tensor[:, rgb_idx], size=(patch_size*4, patch_size*4), mode="nearest")[0].cpu().numpy().transpose(1, 2, 0)))
    axes[0].set_title("1. Raw LR Input (Nearest)")
    axes[0].axis("off")
    
    axes[1].imshow(norm(bic_rgb))
    axes[1].set_title("2. Bicubic (Mathematical Baseline)")
    axes[1].axis("off")
    
    axes[2].imshow(norm(srgan_rgb))
    axes[2].set_title("3. SRGAN (GAN Paper Baseline)")
    axes[2].axis("off")
    
    axes[3].imshow(norm(pred_rgb))
    axes[3].set_title("4. Our 6M Model (Phase 1)")
    axes[3].axis("off")
    
    plt.tight_layout()
    out_file = "/mnt/pune/target_pune_inference.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"Saved {out_file}")

if __name__ == "__main__":
    infer_target()
