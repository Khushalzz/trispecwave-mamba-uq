import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from opensr_srgan import load_inference_model

from metrics.spatial_metrics import psnr, ssim
from metrics.spectral_metrics import spectral_angle_mapper_metric
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from models.detail_head import NAFDetailHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_stage5")

TEACHER_BANDS = [1, 2, 3, 7]
RGB_IN_4BAND = [2, 1, 0]
NIR_IN_4BAND = [3, 2, 1]

def evaluate_stage5(config_path="configs/distill_pune.yaml", samples=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Stage 5 Evaluation on {device}...")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 1. Load backbone
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
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict.get("model_state_dict", state_dict.get("model", state_dict)))
    model.eval()

    # 2. Load Detail Head
    detail_head = NAFDetailHead(in_channels=64, out_channels=4, hidden_dim=64).to(device)
    head_ckpt = "/mnt/pune/checkpoints/detail_head_best.pt"
    if Path(head_ckpt).exists():
        detail_head.load_state_dict(torch.load(head_ckpt, map_location=device, weights_only=True))
    else:
        logger.warning(f"No detail head found at {head_ckpt}, using untrained weights!")
    detail_head.eval()

    # 3. Load Data
    cache_dir = Path(cfg.get("paths", {}).get("teacher_cache_dir", "/mnt/pune/teacher_cache"))
    eval_files = []
    manifest_csv = cache_dir.parent / "patches" / "manifest.csv"
    if manifest_csv.exists():
        import csv
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("split") == "test":
                    tf = cache_dir / f"{r['patch_id']}.pt"
                    if tf.exists():
                        eval_files.append(tf)

    if not eval_files:
        eval_files = sorted(list(cache_dir.glob("*.pt")))
        
    eval_files = eval_files[:samples]
    logger.info(f"Evaluating on {len(eval_files)} cached test patches...")

    metrics = {
        "bicubic": {"psnr": [], "ssim": []},
        "student_base": {"psnr": [], "ssim": []},
        "student_stage5": {"psnr": [], "ssim": []},
    }

    visual_samples = []

    for idx, f in enumerate(eval_files):
        data = torch.load(f, map_location="cpu")
        x = data["input_patch"].unsqueeze(0).to(device).float()
        teacher_hr = data["teacher_output"].unsqueeze(0).to(device).float()
        
        bicubic_rgbn = F.interpolate(x[:, TEACHER_BANDS], size=(256, 256), mode="bicubic", align_corners=False).clamp(0, 1)

        with torch.no_grad():
            aux = model(x, return_aux=True)
            student_base_hr = aux["mean"][:, TEACHER_BANDS].clamp(0, 1)
            high_res = aux["high_res"]
            
            pred_detail = detail_head(high_res)
            student_stage5_hr = (bicubic_rgbn + pred_detail).clamp(0, 1)

        metrics["bicubic"]["psnr"].append(psnr(bicubic_rgbn, teacher_hr).item())
        metrics["bicubic"]["ssim"].append(ssim(bicubic_rgbn, teacher_hr).item())
        
        metrics["student_base"]["psnr"].append(psnr(student_base_hr, teacher_hr).item())
        metrics["student_base"]["ssim"].append(ssim(student_base_hr, teacher_hr).item())

        metrics["student_stage5"]["psnr"].append(psnr(student_stage5_hr, teacher_hr).item())
        metrics["student_stage5"]["ssim"].append(ssim(student_stage5_hr, teacher_hr).item())

        if len(visual_samples) < 6:
            visual_samples.append({
                "patch_id": data.get("patch_id", f"sample_{idx}"),
                "lr": x[:, TEACHER_BANDS].cpu().squeeze(0),
                "bicubic": bicubic_rgbn.cpu().squeeze(0),
                "teacher": teacher_hr.cpu().squeeze(0),
                "base": student_base_hr.cpu().squeeze(0),
                "stage5": student_stage5_hr.cpu().squeeze(0),
                "detail": pred_detail.cpu().squeeze(0)
            })

    # Summary
    logger.info("=== Stage 5 Evaluation ===")
    for model_name in metrics:
        p = np.mean(metrics[model_name]["psnr"])
        s = np.mean(metrics[model_name]["ssim"])
        logger.info(f"{model_name:15s} - PSNR: {p:.2f} dB, SSIM: {s:.4f}")

    # Plot
    outputs_dir = Path("/mnt/pune/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Grid comparison
    n = len(visual_samples)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4.2 * n))
    for i in range(n):
        s = visual_samples[i]
        
        bicubic_rgb = (s["bicubic"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        base_rgb = (s["base"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        stage5_rgb = (s["stage5"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        teacher_rgb = (s["teacher"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        
        teacher_detail = (s["teacher"] - s["bicubic"])[RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5
        pred_detail = s["detail"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5

        ax_row = axes[i] if n > 1 else axes
        ax_row[0].imshow(bicubic_rgb); ax_row[0].set_title(f"Bicubic\n{s['patch_id'][:20]}"); ax_row[0].axis("off")
        ax_row[1].imshow(base_rgb); ax_row[1].set_title("Student Base (32.17 dB)"); ax_row[1].axis("off")
        ax_row[2].imshow(stage5_rgb); ax_row[2].set_title("Student Stage 5 (Detail)"); ax_row[2].axis("off")
        ax_row[3].imshow(teacher_rgb); ax_row[3].set_title("SRGAN Teacher"); ax_row[3].axis("off")
        
        # Visualize absolute detail error
        error = np.abs(pred_detail - teacher_detail).mean(axis=-1)
        ax_row[4].imshow(error, cmap="inferno", vmin=0, vmax=0.15); ax_row[4].set_title("|Pred Detail - True Detail|"); ax_row[4].axis("off")

    plt.tight_layout()
    comparison_path = outputs_dir / "stage5_comparison.png"
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {comparison_path}")

    # Detail map comparison
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    for i in range(n):
        s = visual_samples[i]
        # Normalize details for visualization (shift by 0.5 to center zero)
        teacher_detail = np.clip((s["teacher"] - s["bicubic"])[RGB_IN_4BAND].permute(1, 2, 0).numpy() + 0.5, 0, 1)
        pred_detail = np.clip(s["detail"][RGB_IN_4BAND].permute(1, 2, 0).numpy() + 0.5, 0, 1)

        ax_row = axes[i] if n > 1 else axes
        ax_row[0].imshow(pred_detail); ax_row[0].set_title("Student Pred Detail"); ax_row[0].axis("off")
        ax_row[1].imshow(teacher_detail); ax_row[1].set_title("Teacher True Detail"); ax_row[1].axis("off")
        
    plt.tight_layout()
    detail_path = outputs_dir / "stage5_detail_map.png"
    plt.savefig(detail_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {detail_path}")

if __name__ == "__main__":
    evaluate_stage5()
