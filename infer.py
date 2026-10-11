"""Standalone Inference Suite for TriSpecWave-Mamba-UQ (6.17M Parameter Sentinel-2 4x Super-Resolution).

Runs native 12-band 10m -> 2.5m super-resolution on CPU or CUDA GPU with zero external dependencies.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from models.rgbn_unet_adapter import RGBNUNetAdapter


def load_model(
    checkpoint_path: str = "checkpoints/sen2neon_6m_best.pt",
    config_path: str = "configs/sen2neon_6m.yaml",
    device: str = "cpu",
) -> TriSpecWaveMambaUQ:
    """Load the 6.17M TriSpecWaveMambaUQ model."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = TriSpecWaveMambaUQ(
        feature_dim=cfg.get("feature_dim", 128),
        spectral_dim=cfg.get("spectral_dim", 64),
        naf_depth=cfg.get("naf_depth", 8),
        spatial_depth=cfg.get("spatial_depth", 4),
        band_wavelengths_nm=cfg.get("band_wavelengths_nm"),
        band_gsd_m=cfg.get("band_gsd_m"),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    clean_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)
    model.eval()
    return model


def load_adapter(
    checkpoint_path: str = "checkpoints/sen2neon_unet_adapter_epoch_3.pt",
    device: str = "cpu",
) -> RGBNUNetAdapter | None:
    """Load the optional UNet detailer adapter."""
    if not os.path.exists(checkpoint_path):
        print(f"[!] Optional adapter checkpoint not found at {checkpoint_path}. Continuing with base model.")
        return None

    adapter = RGBNUNetAdapter(in_channels=24, out_channels=4).to(device)
    adapter.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    adapter.eval()
    return adapter


def normalize_for_display(img: np.ndarray, p_min: float = 2.0, p_max: float = 98.0) -> np.ndarray:
    """Percentile stretch for natural satellite RGB/false-color visualization."""
    p_low = np.percentile(img, p_min)
    p_high = np.percentile(img, p_max)
    stretched = np.clip((img - p_low) / max(p_high - p_low, 1e-4), 0.0, 1.0)
    return stretched


def run_inference(
    input_path: str = "data/sample_pune_patch_12b.pt",
    checkpoint_path: str = "checkpoints/sen2neon_6m_best.pt",
    adapter_checkpoint_path: str = "checkpoints/sen2neon_unet_adapter_epoch_3.pt",
    config_path: str = "configs/sen2neon_6m.yaml",
    use_adapter: bool = False,
    device_name: str | None = None,
    output_dir: str = "outputs",
) -> dict:
    """Execute complete 4x super-resolution inference pipeline."""
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65)
    print("[*] TRISPECWAVE-MAMBA-UQ -- 4X MULTI-SPECTRAL SUPER-RESOLUTION")
    print("=" * 65)
    print(f"Device:           {device_name.upper()}")
    print(f"Base Checkpoint:  {checkpoint_path}")
    print(f"Input Patch:      {input_path}")
    print(f"Adapter Enabled:  {use_adapter}")

    # 1. Load Input Data
    if input_path.endswith(".pt"):
        raw_data = torch.load(input_path, map_location="cpu", weights_only=True)
        if isinstance(raw_data, dict):
            lr_tensor = raw_data.get("input_patch", raw_data.get("lr"))
        else:
            lr_tensor = raw_data
    elif input_path.endswith(".npy"):
        lr_np = np.load(input_path)
        lr_tensor = torch.from_numpy(lr_np).float()
    else:
        raise ValueError(f"Unsupported input file format: {input_path}")

    # Ensure shape is [1, 12, H, W] and values are in [0, 1]
    if lr_tensor.ndim == 3:
        lr_tensor = lr_tensor.unsqueeze(0)
    lr_tensor = lr_tensor.float().to(device)
    if lr_tensor.max() > 1.5:
        lr_tensor = lr_tensor / 10000.0

    print(f"Input Tensor:     Shape={tuple(lr_tensor.shape)} (12 bands @ 10m/20m/60m GSD)")

    # 2. Load Base Model
    t0 = time.time()
    model = load_model(checkpoint_path=checkpoint_path, config_path=config_path, device=device_name)
    load_time = time.time() - t0
    print(f"Model Loaded in:  {load_time:.2f}s (6.17M Parameters)")

    # 3. Load Adapter if requested
    adapter = None
    if use_adapter:
        adapter = load_adapter(checkpoint_path=adapter_checkpoint_path, device=device_name)

    # 4. Perform Inference
    with torch.no_grad():
        # Warmup pass
        _ = model(lr_tensor[:1, :, :32, :32]) if lr_tensor.shape[2] > 32 else None

        t_infer_start = time.time()
        output = model(lr_tensor)
        infer_latency_ms = (time.time() - t_infer_start) * 1000.0

        if isinstance(output, tuple):
            pred_12b, logvar = output
            sigma = torch.exp(0.5 * logvar)
        else:
            pred_12b = output
            sigma = torch.zeros_like(pred_12b)

        pred_12b = pred_12b.clamp(0.0, 1.0)

        # Baseline Bicubic Interpolation
        lr_bicubic = F.interpolate(lr_tensor, scale_factor=4, mode="bicubic", align_corners=False).clamp(0.0, 1.0)

        # Apply UNet Adapter detailer if enabled
        if adapter is not None:
            adapter_rgbn = adapter(pred_12b, lr_bicubic)
            # Replace RGBN bands (1=B2, 2=B3, 3=B4, 7=B8) with adapter refined output
            pred_12b[:, [1, 2, 3, 7]] = adapter_rgbn

    print(f"Output Tensor:    Shape={tuple(pred_12b.shape)} (12 bands @ 2.5m GSD)")
    print(f"Inference Time:   {infer_latency_ms:.2f} ms")

    # 5. Extract Bands for Visualizations
    # Sentinel-2 Band Indices: 1=B2(Blue), 2=B3(Green), 3=B4(Red), 7=B8(NIR)
    b_red, b_green, b_blue, b_nir = 3, 2, 1, 7

    os.makedirs(output_dir, exist_ok=True)

    # Low-Res Bicubic RGB
    bicubic_rgb = lr_bicubic[0, [b_red, b_green, b_blue]].cpu().numpy().transpose(1, 2, 0)
    # Model 2.5m RGB
    sr_rgb = pred_12b[0, [b_red, b_green, b_blue]].cpu().numpy().transpose(1, 2, 0)
    # Model 2.5m Color Infrared (CIR): NIR(B8), Red(B4), Green(B3)
    sr_cir = pred_12b[0, [b_nir, b_red, b_green]].cpu().numpy().transpose(1, 2, 0)
    # Normalized Difference Vegetation Index (NDVI): (NIR - Red) / (NIR + Red)
    nir_band = pred_12b[0, b_nir].cpu().numpy()
    red_band = pred_12b[0, b_red].cpu().numpy()
    ndvi = (nir_band - red_band) / np.maximum(nir_band + red_band, 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Uncertainty map (RGB mean uncertainty)
    unc_map = sigma[0, [b_red, b_green, b_blue]].mean(dim=0).cpu().numpy()

    # Save individual outputs
    vis_bicubic = normalize_for_display(bicubic_rgb)
    vis_sr_rgb = normalize_for_display(sr_rgb)
    vis_sr_cir = normalize_for_display(sr_cir)

    # 6. Save Side-by-Side Comparison Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    axes[0].imshow(vis_bicubic)
    axes[0].set_title("10m Native (Bicubic 4x)", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    model_title = "TriSpecWave 2.5m (+ UNet)" if adapter is not None else "TriSpecWave-Mamba-UQ 2.5m"
    axes[1].imshow(vis_sr_rgb)
    axes[1].set_title(f"{model_title} (True Color)", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(vis_sr_cir)
    axes[2].set_title("2.5m Color Infrared (CIR)", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    im_unc = axes[3].imshow(unc_map, cmap="viridis")
    axes[3].set_title("Aleatoric Uncertainty (σ)", fontsize=13, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im_unc, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "sample_comparison.png")
    plt.savefig(comparison_path, bbox_inches="tight")
    plt.close()

    print("-" * 65)
    print("[SUCCESS] INFERENCE COMPLETED SUCCESSFULLY")
    print(f"Saved Side-by-Side Plot:  {comparison_path}")
    print("=" * 65)

    return {
        "latency_ms": infer_latency_ms,
        "comparison_plot": comparison_path,
        "sr_rgb_shape": sr_rgb.shape,
    }


def main():
    parser = argparse.ArgumentParser(description="TriSpecWave-Mamba-UQ 4x Super-Resolution Inference")
    parser.add_argument("--input", type=str, default="data/sample_pune_patch_12b.pt", help="Path to input 12-band patch (.pt or .npy)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sen2neon_6m_best.pt", help="Base model checkpoint")
    parser.add_argument("--adapter-checkpoint", type=str, default="checkpoints/sen2neon_unet_adapter_epoch_3.pt", help="UNet detailer checkpoint")
    parser.add_argument("--config", type=str, default="configs/sen2neon_6m.yaml", help="Model config YAML")
    parser.add_argument("--use-adapter", action="store_true", help="Enable optional UNet detailer for RGB-NIR bands")
    parser.add_argument("--device", type=str, default=None, help="Target device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for visual results")
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        adapter_checkpoint_path=args.adapter_checkpoint,
        config_path=args.config,
        use_adapter=args.use_adapter,
        device_name=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
