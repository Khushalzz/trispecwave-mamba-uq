"""Train TriSpecWave-Mamba-UQ student distilled from OpenSR-SRGAN teacher.

Uses 12 Sentinel-2 L2A bands, supervises RGB-NIR (B2, B3, B4, B8) with cached teacher targets,
enforces 12-band sensor consistency via MTF/PSF degradation, and guides spectral angles with SAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from losses.mtf_psf_loss import MTFPSFLoss
from losses.spectral_consistency_loss import spectral_angle_mapper
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distill_srgan")

TEACHER_BANDS = [1, 2, 3, 7]  # B2, B3, B4, B8


class TeacherCacheDataset(Dataset):
    """Loads pre-cached (input_patch, teacher_output) pairs into RAM for zero-latency training."""

    def __init__(self, cache_dir: str | Path, split: str = "train", max_samples: int | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        
        # 1. Gather paths
        sample_paths = []
        manifest_csv = self.cache_dir.parent / "patches" / "manifest.csv"

        if manifest_csv.exists():
            with open(manifest_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("split") == split:
                        sample_paths.append(self.cache_dir / f"{r['patch_id']}.pt")
        else:
            sample_paths = sorted(list(self.cache_dir.glob("*.pt")))

        if max_samples and max_samples < len(sample_paths):
            sample_paths = sample_paths[:max_samples]

        logger.info(f"Dataset split='{split}': loading {len(sample_paths)} patches into RAM from {cache_dir}...")

        # 2. Preload into memory
        self.data = []
        for f in sample_paths:
            pt = torch.load(f, map_location="cpu", weights_only=False)
            x = pt["input_patch"].float().clone()
            y = pt["teacher_output"].float().clone()
            self.data.append((x, y))
            
        logger.info(f"Dataset split='{split}': finished loading {len(self.data)} patches into RAM.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distill_student(
    config_path: str = "configs/distill_pune.yaml",
    max_steps: int | None = None,
    num_patches: int | None = None,
    resume: bool = False,
    warm_start: bool = True,
) -> dict[str, Any]:
    """Execute remote distillation training loop on GPU."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Distillation training running on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    paths = cfg.get("paths", {})
    cache_dir = Path(paths.get("teacher_cache_dir", "/mnt/pune/teacher_cache"))
    ckpt_dir = Path(paths.get("checkpoints_dir", "/mnt/pune/checkpoints"))
    logs_dir = Path(paths.get("logs_dir", "/mnt/pune/logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "distill.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    # 1. Datasets and DataLoaders
    train_ds = TeacherCacheDataset(cache_dir, split="train", max_samples=num_patches)
    val_ds = TeacherCacheDataset(cache_dir, split="val", max_samples=100)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training patches found in {cache_dir}. Run cache_teacher_targets first.")

    batch_size = cfg.get("batch_size", 16)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    ) if len(val_ds) > 0 else None

    # 2. Build Student Model
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
    ).to(device)

    total_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info(f"Initialized TriSpecWaveMambaUQ: {total_params:,} trainable parameters.")

    # 3. Loss Modules
    mtf_loss_module = MTFPSFLoss(scale=cfg.get("scale", 4), kernel_size=5, sigma=1.2).to(device)
    loss_weights = cfg.get("loss_weights", {"teacher": 1.0, "sensor": 0.25, "sam": 0.05})
    w_teacher = loss_weights.get("teacher", 1.0)
    w_sensor = loss_weights.get("sensor", 0.25)
    w_sam = loss_weights.get("sam", 0.05)

    # 4. Optimizer and Scaler
    lr = float(cfg.get("learning_rate", 5e-5))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("mixed_precision", True) and torch.cuda.is_available())

    total_steps = max_steps or cfg.get("max_steps", 2500)
    # Subagent recommendation: Learning Rate Warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        div_factor=25,
        final_div_factor=100,
    )

    # Resume or Warm Start from best.pt
    start_step = 0
    best_val_loss = float("inf")
    last_ckpt_path = ckpt_dir / "last.pt"
    best_ckpt_path = ckpt_dir / "best.pt"
    
    # Path to the original converged L1 checkpoint
    original_best_path = Path("/mnt/pune/checkpoints/best.pt")

    if resume and last_ckpt_path.exists():
        try:
            ckpt = torch.load(last_ckpt_path, map_location=device)
            student.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt.get("step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            logger.info(f"Resumed from step {start_step} with best_val_loss={best_val_loss:.4f}")
        except Exception as e:
            logger.warning(f"Failed to resume checkpoint: {e}")
    elif warm_start and (best_ckpt_path.exists() or original_best_path.exists()):
        load_path = best_ckpt_path if best_ckpt_path.exists() else original_best_path
        try:
            ckpt = torch.load(load_path, map_location=device)
            student.load_state_dict(ckpt["model"], strict=False)
            prev_val = ckpt.get("best_val_loss", float("inf"))
            logger.info(f"Warm started fresh optimizer from {load_path} (previous best Val = {prev_val:.4f})")
        except Exception as e:
            logger.warning(f"Failed to warm start: {e}")

    log_every = cfg.get("log_every", 50)
    val_every = cfg.get("val_every", 200)
    save_every = cfg.get("save_every", 1000)

    logger.info(f"Starting distillation training: step {start_step} -> {total_steps} (initial lr={lr:.6f})...")
    student.train()
    step = start_step
    train_iter = iter(train_loader)
    start_time = time.time()

    running_teacher_loss = 0.0
    running_sensor_loss = 0.0
    running_sam_loss = 0.0
    running_total_loss = 0.0

    def charbonnier_loss(pred, target, eps=1e-6):
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps))

    def gradient_loss(pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    while step < total_steps:
        step += 1
        try:
            x, teacher_hr = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, teacher_hr = next(train_iter)

        x = x.to(device, non_blocking=True)  # [B, 12, 64, 64]
        teacher_hr = teacher_hr.to(device, non_blocking=True)  # [B, 4, 256, 256]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=cfg.get("mixed_precision", True) and torch.cuda.is_available()):
            student_hr, logvar = student(x)  # [B, 12, 256, 256]
            student_rgbn = student_hr[:, TEACHER_BANDS]  # [B, 4, 256, 256]

            # 1. Distillation Edge-Aware loss against teacher RGB-NIR
            l_charb = charbonnier_loss(student_rgbn, teacher_hr)
            l_grad = gradient_loss(student_rgbn, teacher_hr)
            loss_teacher = l_charb + 0.20 * l_grad

            # 2. Sensor consistency loss via MTF degradation on all 12 bands
            degraded_12b = mtf_loss_module.degrade(student_hr, output_size=x.shape[-2:])
            loss_sensor = F.l1_loss(degraded_12b, x)

            # 3. Spectral angle mapper loss on RGB-NIR
            loss_sam = spectral_angle_mapper(student_rgbn, teacher_hr)

            # Total loss
            total_loss = (
                w_teacher * loss_teacher
                + w_sensor * loss_sensor
                + w_sam * loss_sam
            )

        scaler.scale(total_loss).backward()
        if cfg.get("grad_clip_norm"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["grad_clip_norm"])
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_teacher_loss += loss_teacher.item()
        running_sensor_loss += loss_sensor.item()
        running_sam_loss += loss_sam.item()
        running_total_loss += total_loss.item()

        # Logging
        if step % log_every == 0 or step == total_steps:
            elapsed = time.time() - start_time
            steps_per_sec = (step - start_step) / max(elapsed, 1e-4)
            avg_teacher = running_teacher_loss / log_every
            avg_sensor = running_sensor_loss / log_every
            avg_sam = running_sam_loss / log_every
            avg_total = running_total_loss / log_every
            curr_lr = scheduler.get_last_lr()[0]

            logger.info(
                f"Step [{step}/{total_steps}] "
                f"Total: {avg_total:.4f} | "
                f"Teacher L1: {avg_teacher:.4f} | "
                f"Sensor L1: {avg_sensor:.4f} | "
                f"SAM: {avg_sam:.4f} rad | "
                f"lr: {curr_lr:.2e} | "
                f"Speed: {steps_per_sec:.1f} steps/s"
            )
            running_teacher_loss = 0.0
            running_sensor_loss = 0.0
            running_sam_loss = 0.0
            running_total_loss = 0.0

        # Periodic Validation
        if val_loader and (step % val_every == 0 or step == total_steps):
            student.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for vx, v_teacher in val_loader:
                    vx = vx.to(device)
                    v_teacher = v_teacher.to(device)
                    v_hr, _ = student(vx)
                    v_rgbn = v_hr[:, TEACHER_BANDS]
                    l_t = F.l1_loss(v_rgbn, v_teacher)
                    val_loss += l_t.item()
                    val_count += 1
            student.train()
            mean_val_loss = val_loss / max(val_count, 1)
            logger.info(f"--- Validation Step {step}: Val Teacher L1 = {mean_val_loss:.4f} ---")

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_ckpt = ckpt_dir / "best.pt"
                torch.save(
                    {
                        "step": step,
                        "model": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "config": cfg,
                    },
                    best_ckpt,
                )
                logger.info(f"Saved new best checkpoint to {best_ckpt} (Val L1 = {best_val_loss:.4f})")

        # Save checkpoint periodically
        if step % save_every == 0 or step == total_steps:
            torch.save(
                {
                    "step": step,
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                },
                last_ckpt_path,
            )

    train_duration = time.time() - start_time
    logger.info(f"Distillation finished in {train_duration:.1f}s. Checkpoints saved to {ckpt_dir}.")

    return {
        "final_step": step,
        "best_val_loss": best_val_loss,
        "total_params": total_params,
        "train_duration_sec": train_duration,
        "best_checkpoint": str(ckpt_dir / "best.pt"),
        "last_checkpoint": str(last_ckpt_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distil OpenSR-SRGAN to TriSpecWave-Mamba-UQ")
    parser.add_argument("--config", default="configs/distill_pune.yaml", help="Path to config")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--num-patches", type=int, default=None, help="Limit number of training patches")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint and optimizer state")
    parser.add_argument("--no-warm-start", action="store_true", help="Train from scratch without loading best.pt weights")
    args = parser.parse_args()

    distill_student(
        config_path=args.config,
        max_steps=args.max_steps,
        num_patches=args.num_patches,
        resume=args.resume,
        warm_start=not args.no_warm_start,
    )

