"""Evaluation suite comparing Bicubic baseline, SRGAN teacher, and TriSpecWave-Mamba-UQ student.

Computes PSNR, SSIM, SAM, NDVI consistency, per-band error, and benchmarks inference latency
and GPU VRAM. Generates comparison visualizations and saves metrics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from opensr_srgan import load_inference_model

from metrics.spatial_metrics import psnr, ssim
from metrics.spectral_metrics import spectral_angle_mapper_metric
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_distill")

TEACHER_BANDS = [1, 2, 3, 7]  # B2, B3, B4, B8 in 12-band order
# In the 4-band RGB-NIR array (B2, B3, B4, B8):
# Index 0: B2 (Blue), Index 1: B3 (Green), Index 2: B4 (Red), Index 3: B8 (NIR)
RGB_IN_4BAND = [2, 1, 0]  # Red, Green, Blue for true-color plotting
NIR_IN_4BAND = [3, 2, 1]  # NIR, Red, Green for false-color vegetation plotting


def compute_ndvi(tensor_4b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    nir = tensor_4b[:, 3:4]
    red = tensor_4b[:, 2:3]
    return (nir - red) / (nir + red).abs().clamp_min(eps)


def evaluate_model(
    config_path: str = "configs/distill_pune.yaml",
    checkpoint_path: str | None = None,
    num_eval_samples: int = 50,
    split: str = "test",
) -> dict[str, Any]:
    """Run full evaluation suite and generate reports/plots."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running evaluation on {device} (split='{split}')...")

    paths = cfg.get("paths", {})
    cache_dir = Path(paths.get("teacher_cache_dir", "/mnt/pune/teacher_cache"))
    ckpt_dir = Path(paths.get("checkpoints_dir", "/mnt/pune/checkpoints"))
    metrics_dir = Path(paths.get("metrics_dir", "/mnt/pune/metrics"))
    outputs_dir = Path(paths.get("outputs_dir", "/mnt/pune/outputs"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Student Checkpoint
    ckpt_file = Path(checkpoint_path) if checkpoint_path else (ckpt_dir / "best.pt")
    if not ckpt_file.exists():
        ckpt_file = ckpt_dir / "last.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_file}")

    logger.info(f"Loading student model checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=device)

    student = TriSpecWaveMambaUQ(
        in_channels=cfg.get("in_channels", 12),
        out_channels=cfg.get("out_channels", 12),
        feature_dim=cfg.get("feature_dim", 64),
        spectral_dim=cfg.get("spectral_dim", 64),
        naf_depth=cfg.get("naf_depth", 4),
        spatial_depth=cfg.get("spatial_depth", 2),
        graph_heads=cfg.get("graph_heads", 4),
        dropout=cfg.get("dropout", 0.1),
        band_wavelengths_nm=cfg.get("band_wavelengths_nm"),
        band_gsd_m=cfg.get("band_gsd_m"),
        upsample_stages=cfg.get("upsample_stages", 2),
    ).to(device).eval()

    student.load_state_dict(ckpt["model"])
    student_params = sum(p.numel() for p in student.parameters())

    # 2. Load Frozen Teacher for latency comparison
    logger.info("Loading OpenSR-SRGAN teacher for latency comparison...")
    teacher = load_inference_model("RGB-NIR").to(device).eval()

    # 3. Load Evaluation Data
    eval_files = []
    manifest_csv = cache_dir.parent / "patches" / "manifest.csv"
    if manifest_csv.exists():
        import csv
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if split and r.get("split") == split:
                    tf = cache_dir / f"{r['patch_id']}.pt"
                    if tf.exists():
                        eval_files.append(tf)
                elif not split and r.get("split") in ("val", "test"):
                    tf = cache_dir / f"{r['patch_id']}.pt"
                    if tf.exists():
                        eval_files.append(tf)

    if not eval_files:
        logger.warning(f"No cached patches found specifically for split='{split}'. Trying val split or all cached...")
        if manifest_csv.exists():
            import csv
            with open(manifest_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("split") in ("val", "test"):
                        tf = cache_dir / f"{r['patch_id']}.pt"
                        if tf.exists():
                            eval_files.append(tf)

    if not eval_files:
        cache_files = sorted(list(cache_dir.glob("*.pt")))
        eval_files = cache_files[:num_eval_samples]
    else:
        eval_files = eval_files[:num_eval_samples]

    logger.info(f"Evaluating on {len(eval_files)} cached evaluation patches (split='{split}')...")

    # Metrics accumulators
    metrics_bicubic = {"psnr": [], "ssim": [], "sam": [], "ndvi_mae": []}
    metrics_student = {"psnr": [], "ssim": [], "sam": [], "ndvi_mae": []}
    per_band_l1 = {b: [] for b in ["B2", "B3", "B4", "B8"]}

    # Timing benchmarks
    teacher_times = []
    student_times = []

    visual_samples = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for idx, f in enumerate(eval_files):
        data = torch.load(f, map_location="cpu")
        x = data["input_patch"].unsqueeze(0).to(device).float()  # [1, 12, 64, 64]
        teacher_hr = data["teacher_output"].unsqueeze(0).to(device).float()  # [1, 4, 256, 256]
        x_rgbn = x[:, TEACHER_BANDS]  # [1, 4, 64, 64]

        # Bicubic baseline (4x upsample)
        bicubic_hr = F.interpolate(x_rgbn, size=(256, 256), mode="bicubic", align_corners=False).clamp(0.0, 1.0)

        # Student inference benchmark
        t0 = time.perf_counter()
        with torch.no_grad():
            student_hr, logvar = student(x)
            student_rgbn = student_hr[:, TEACHER_BANDS].clamp(0.0, 1.0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        student_times.append(t1 - t0)

        # Teacher inference benchmark
        t2 = time.perf_counter()
        with torch.no_grad():
            _ = teacher.predict_step(x_rgbn)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        t3 = time.perf_counter()
        teacher_times.append(t3 - t2)

        # Compute Metrics against Teacher Pseudo-Target
        # Bicubic
        b_p = psnr(bicubic_hr, teacher_hr).item()
        b_s = ssim(bicubic_hr, teacher_hr).item()
        b_sam = spectral_angle_mapper_metric(bicubic_hr, teacher_hr).item()
        b_ndvi = F.l1_loss(compute_ndvi(bicubic_hr), compute_ndvi(teacher_hr)).item()
        metrics_bicubic["psnr"].append(b_p)
        metrics_bicubic["ssim"].append(b_s)
        metrics_bicubic["sam"].append(b_sam)
        metrics_bicubic["ndvi_mae"].append(b_ndvi)

        # Student
        s_p = psnr(student_rgbn, teacher_hr).item()
        s_s = ssim(student_rgbn, teacher_hr).item()
        s_sam = spectral_angle_mapper_metric(student_rgbn, teacher_hr).item()
        s_ndvi = F.l1_loss(compute_ndvi(student_rgbn), compute_ndvi(teacher_hr)).item()
        metrics_student["psnr"].append(s_p)
        metrics_student["ssim"].append(s_s)
        metrics_student["sam"].append(s_sam)
        metrics_student["ndvi_mae"].append(s_ndvi)

        # Per-band L1 error
        for bi, bname in enumerate(["B2", "B3", "B4", "B8"]):
            per_band_l1[bname].append(F.l1_loss(student_rgbn[:, bi], teacher_hr[:, bi]).item())

        if len(visual_samples) < 6:
            visual_samples.append(
                {
                    "lr": x_rgbn.cpu().squeeze(0),
                    "bicubic": bicubic_hr.cpu().squeeze(0),
                    "teacher": teacher_hr.cpu().squeeze(0),
                    "student": student_rgbn.cpu().squeeze(0),
                    "student_12b": student_hr.cpu().squeeze(0),
                    "logvar": logvar.cpu().squeeze(0),
                    "patch_id": data.get("patch_id", f"sample_{idx}"),
                }
            )

    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

    # Aggregate results
    report = {
        "evaluation_summary": {
            "evaluation_patch_count": len(eval_files),
            "student_parameter_count": student_params,
            "gpu_peak_vram_mb": round(peak_vram_mb, 2),
            "avg_teacher_inference_ms": round(float(np.mean(teacher_times) * 1000), 2),
            "avg_student_inference_ms": round(float(np.mean(student_times) * 1000), 2),
            "known_bicubic_baseline_reference_db": 35.38,
        },
        "labels_and_disclaimer": {
            "srgan_status": "Pseudo-target (pretrained RGB-NIR SRGAN teacher supervision)",
            "student_12band_status": "Model-estimated surface reflectance",
            "super_resolution_nature": "Super-resolved prediction (10m -> 2.5m progressive), not guaranteed physical 2.5m measurement",
        },
        "bicubic_baseline_vs_teacher": {
            "mean_psnr_db": round(float(np.mean(metrics_bicubic["psnr"])), 2),
            "mean_ssim": round(float(np.mean(metrics_bicubic["ssim"])), 4),
            "mean_sam_rad": round(float(np.mean(metrics_bicubic["sam"])), 4),
            "mean_ndvi_mae": round(float(np.mean(metrics_bicubic["ndvi_mae"])), 4),
        },
        "tri_spec_wave_student_vs_teacher": {
            "mean_psnr_db": round(float(np.mean(metrics_student["psnr"])), 2),
            "mean_ssim": round(float(np.mean(metrics_student["ssim"])), 4),
            "mean_sam_rad": round(float(np.mean(metrics_student["sam"])), 4),
            "mean_ndvi_mae": round(float(np.mean(metrics_student["ndvi_mae"])), 4),
        },
        "per_band_student_mae": {b: round(float(np.mean(per_band_l1[b])), 5) for b in per_band_l1},
    }

    metrics_json_path = metrics_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=== Distillation Evaluation Results ===")
    logger.info(f"Student Parameters:          {student_params:,}")
    logger.info(f"Student Inference Latency:   {report['evaluation_summary']['avg_student_inference_ms']} ms")
    logger.info(f"Teacher Inference Latency:   {report['evaluation_summary']['avg_teacher_inference_ms']} ms")
    logger.info(f"Bicubic Baseline PSNR:       {report['bicubic_baseline_vs_teacher']['mean_psnr_db']} dB")
    logger.info(f"Student Distilled PSNR:      {report['tri_spec_wave_student_vs_teacher']['mean_psnr_db']} dB")
    logger.info(f"Student Distilled SSIM:      {report['tri_spec_wave_student_vs_teacher']['mean_ssim']}")
    logger.info(f"Student Distilled SAM:       {report['tri_spec_wave_student_vs_teacher']['mean_sam_rad']} rad")
    logger.info(f"Per-band MAE: {report['per_band_student_mae']}")
    logger.info("=======================================")

    # 4. Generate Visual Comparisons
    _generate_comparison_plots(visual_samples, outputs_dir)

    return report


def _generate_comparison_plots(samples: list[dict[str, Any]], outputs_dir: Path) -> None:
    """Generate high-resolution PNG comparison figures."""
    if not samples:
        return

    # 1. Main multi-patch grid comparison
    n = min(6, len(samples))
    fig, axes = plt.subplots(n, 5, figsize=(20, 4.2 * n))

    for i in range(n):
        s = samples[i]
        # Stretch reflectance for clear visualization (scale by 3.5, clamp to [0, 1])
        lr_rgb = (s["lr"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        bicubic_rgb = (s["bicubic"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        teacher_rgb = (s["teacher"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        student_rgb = (s["student"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        residual = np.abs(student_rgb - teacher_rgb).mean(axis=-1)  # [256, 256]

        ax_row = axes[i] if n > 1 else axes
        ax_row[0].imshow(lr_rgb)
        ax_row[0].set_title(f"LR 10m Input (64x64)\n{s['patch_id'][:20]}")
        ax_row[0].axis("off")

        ax_row[1].imshow(bicubic_rgb)
        ax_row[1].set_title("Bicubic 4x Baseline\n(2.5m, 256x256)")
        ax_row[1].axis("off")

        ax_row[2].imshow(student_rgb)
        ax_row[2].set_title("TriSpecWave-Mamba-UQ\n(Student 2.5m Edge-Tuned)")
        ax_row[2].axis("off")

        ax_row[3].imshow(teacher_rgb)
        ax_row[3].set_title("SRGAN Teacher\n(Pseudo-Target 2.5m)")
        ax_row[3].axis("off")

        im = ax_row[4].imshow(residual, cmap="inferno", vmin=0, vmax=0.15)
        ax_row[4].set_title("|Student - Teacher|\nResidual Error")
        ax_row[4].axis("off")

    plt.tight_layout()
    comparison_path = outputs_dir / "pune_comparison.png"
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved visual comparison to {comparison_path}")

    # 2. Individual high-res patch comparisons (for direct inspection)
    for i in range(min(4, len(samples))):
        s = samples[i]
        lr_rgb = (s["lr"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        bicubic_rgb = (s["bicubic"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        student_rgb = (s["student"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        teacher_rgb = (s["teacher"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        residual = np.abs(student_rgb - teacher_rgb).mean(axis=-1)

        st_nir = (s["student"][NIR_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        tc_nir = (s["teacher"][NIR_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)

        p_fig, p_axes = plt.subplots(2, 4, figsize=(18, 9))
        
        # Row 1: True Color (RGB)
        p_axes[0, 0].imshow(lr_rgb); p_axes[0, 0].set_title(f"LR 10m Input\n({s['patch_id']})"); p_axes[0, 0].axis("off")
        p_axes[0, 1].imshow(bicubic_rgb); p_axes[0, 1].set_title("Bicubic 4x Baseline (2.5m)"); p_axes[0, 1].axis("off")
        p_axes[0, 2].imshow(student_rgb); p_axes[0, 2].set_title("TriSpecWave Student (2.5m)"); p_axes[0, 2].axis("off")
        p_axes[0, 3].imshow(teacher_rgb); p_axes[0, 3].set_title("SRGAN Teacher Target (2.5m)"); p_axes[0, 3].axis("off")

        # Row 2: False Color NIR & Error & Uncertainty
        std_map = torch.exp(0.5 * s["logvar"]).mean(dim=0).numpy()
        p_axes[1, 0].imshow(st_nir); p_axes[1, 0].set_title("Student False Color (NIR-R-G)"); p_axes[1, 0].axis("off")
        p_axes[1, 1].imshow(tc_nir); p_axes[1, 1].set_title("Teacher False Color (NIR-R-G)"); p_axes[1, 1].axis("off")
        im_res = p_axes[1, 2].imshow(residual, cmap="inferno", vmin=0, vmax=0.15)
        p_axes[1, 2].set_title("|Student - Teacher| Error"); p_axes[1, 2].axis("off")
        p_fig.colorbar(im_res, ax=p_axes[1, 2], fraction=0.046, pad=0.04)

        im_unc = p_axes[1, 3].imshow(std_map, cmap="viridis")
        p_axes[1, 3].set_title("Student Aleatoric Uncertainty"); p_axes[1, 3].axis("off")
        p_fig.colorbar(im_unc, ax=p_axes[1, 3], fraction=0.046, pad=0.04)

        p_fig.tight_layout()
        indiv_path = outputs_dir / f"unseen_patch_{i+1}.png"
        p_fig.savefig(indiv_path, dpi=200, bbox_inches="tight")
        plt.close(p_fig)
        logger.info(f"Saved individual patch figure to {indiv_path}")

    # 3. Student RGB-NIR detailed plot: True Color + False Color (NIR)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    for i in range(n):
        s = samples[i]
        st_rgb = (s["student"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        st_nir = (s["student"][NIR_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)

        ax_row = axes[i] if n > 1 else axes
        ax_row[0].imshow(st_rgb)
        ax_row[0].set_title(f"Student 2.5m True Color (RGB)\n{s['patch_id'][:20]}")
        ax_row[0].axis("off")

        ax_row[1].imshow(st_nir)
        ax_row[1].set_title("Student 2.5m False Color (NIR-R-G)")
        ax_row[1].axis("off")

    plt.tight_layout()
    rgbn_path = outputs_dir / "pune_student_rgbn.png"
    plt.savefig(rgbn_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved student RGBN image to {rgbn_path}")

    # 4. Uncertainty map plot
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    for i in range(n):
        s = samples[i]
        st_rgb = (s["student"][RGB_IN_4BAND].permute(1, 2, 0).numpy() * 3.5).clip(0, 1)
        std_map = torch.exp(0.5 * s["logvar"]).mean(dim=0).numpy()

        ax_row = axes[i] if n > 1 else axes
        ax_row[0].imshow(st_rgb)
        ax_row[0].set_title(f"Super-Resolved Prediction\n{s['patch_id'][:20]}")
        ax_row[0].axis("off")

        im_u = ax_row[1].imshow(std_map, cmap="viridis")
        ax_row[1].set_title("Predicted Aleatoric Uncertainty\n(Std Dev sigma)")
        ax_row[1].axis("off")
        fig.colorbar(im_u, ax=ax_row[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    uncertainty_path = outputs_dir / "pune_uncertainty.png"
    plt.savefig(uncertainty_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved uncertainty map to {uncertainty_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate student distillation")
    parser.add_argument("--config", default="configs/distill_pune.yaml", help="Path to config")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint file")
    parser.add_argument("--samples", type=int, default=50, help="Number of eval samples")
    parser.add_argument("--split", default="test", help="Data split to evaluate on (test, val, or all)")
    args = parser.parse_args()

    evaluate_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_eval_samples=args.samples,
        split=args.split,
    )
