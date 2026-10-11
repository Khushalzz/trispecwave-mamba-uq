"""Cache OpenSR-SRGAN RGB-NIR teacher targets on Modal GPU.

Loads the pretrained ESA OpenSR-SRGAN RGB-NIR teacher, runs predict_step on bands
[1, 2, 3, 7] (B2, B3, B4, B8), and saves 4x super-resolved targets to /mnt/pune/teacher_cache/.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from opensr_srgan import load_inference_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cache_teacher")

TEACHER_BANDS = [1, 2, 3, 7]  # B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)


def cache_teacher_targets(
    patches_dir: str = "/mnt/pune/patches",
    cache_dir: str = "/mnt/pune/teacher_cache",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_patches: int = 500,
    batch_size: int = 8,
) -> int:
    """Run frozen OpenSR-SRGAN teacher and cache super-resolved RGB-NIR targets."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest_csv = Path(patches_dir) / "manifest.csv"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Patches manifest not found at {manifest_csv}")

    rows = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("Manifest is empty")

    logger.info(f"Loaded {len(rows)} entries from manifest. Target cache: {cache_dir}")
    rows = rows[:max_patches]

    # Load frozen teacher model
    logger.info(f"Loading OpenSR-SRGAN RGB-NIR teacher onto {device}...")
    teacher = load_inference_model("RGB-NIR").to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Pre-flight single patch verification
    sample_pt = torch.load(rows[0]["input_path"], map_location="cpu")
    sample_x = sample_pt["input"].unsqueeze(0).to(device)  # [1, 12, 64, 64]
    sample_teacher_in = sample_x[:, TEACHER_BANDS]  # [1, 4, 64, 64]

    with torch.no_grad():
        sample_teacher_out = teacher.predict_step(sample_teacher_in)

    logger.info("=== Pre-flight Teacher Shape Verification ===")
    logger.info(f"Input student patch shape:  {list(sample_x.shape)}")
    logger.info(f"Teacher input shape:        {list(sample_teacher_in.shape)}")
    logger.info(f"Teacher output shape:       {list(sample_teacher_out.shape)}")
    logger.info(
        f"Teacher output range:       [{sample_teacher_out.min().item():.4f}, {sample_teacher_out.max().item():.4f}]"
    )
    logger.info("=============================================")

    cached_count = 0
    skipped_count = 0

    # Batch process patches for speed
    pending_items = []
    for row in rows:
        patch_id = row["patch_id"]
        out_target_path = cache_path / f"{patch_id}.pt"
        if out_target_path.exists() and out_target_path.stat().st_size > 5000:
            skipped_count += 1
            continue
        pending_items.append((row, out_target_path))

    logger.info(f"Already cached: {skipped_count}. Pending: {len(pending_items)}.")

    from concurrent.futures import ThreadPoolExecutor

    total_batches = (len(pending_items) + batch_size - 1) // batch_size if pending_items else 0
    logger.info(f"Processing {len(pending_items)} pending targets in {total_batches} batches (batch_size={batch_size}) with 8 worker threads...")

    def _save_one(item_tuple):
        payload, out_path = item_tuple
        torch.save(payload, out_path)

    for b_idx, i in enumerate(range(0, len(pending_items), batch_size)):
        batch_slice = pending_items[i : i + batch_size]

        # Parallel load input patches from volume
        def _load_input(item):
            row, out_path = item
            pt = torch.load(row["input_path"], map_location="cpu")
            return pt["input"].clone().contiguous(), row, out_path, pt.get("metadata", {})

        with ThreadPoolExecutor(max_workers=8) as executor:
            loaded_items = list(executor.map(_load_input, batch_slice))

        batch_tensors = [x[0] for x in loaded_items]
        batch_metas = [(x[1], x[2], x[3]) for x in loaded_items]

        batch_stack = torch.stack(batch_tensors, dim=0).to(device)  # [B, 12, 64, 64]
        teacher_in = batch_stack[:, TEACHER_BANDS]  # [B, 4, 64, 64]

        with torch.no_grad():
            teacher_out = teacher.predict_step(teacher_in)  # [B, 4, 256, 256]

        teacher_out_cpu = teacher_out.cpu()

        save_items = []
        for j, (row, out_path, meta) in enumerate(batch_metas):
            save_payload = {
                "input_patch": batch_tensors[j].clone().contiguous(),  # [12, 64, 64]
                "teacher_output": teacher_out_cpu[j].clone().contiguous(),  # [4, 256, 256]
                "scene_id": row["scene_id"],
                "acquisition_date": row["acquisition_date"],
                "patch_id": row["patch_id"],
                "cloud_fraction": float(row["cloud_fraction"]),
                "transform": meta.get("transform", []),
                "crs": row["crs"],
                "split": row["split"],
            }
            save_items.append((save_payload, out_path))

        # Parallel save to volume
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(_save_one, save_items))

        cached_count += len(batch_slice)
        logger.info(f"Batch [{b_idx + 1}/{total_batches}]: cached {cached_count}/{len(pending_items)} targets (Total cached: {cached_count + skipped_count}/{len(rows)})")

    logger.info(f"Teacher target caching complete. Total cached: {cached_count + skipped_count}.")
    return cached_count + skipped_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache OpenSR-SRGAN teacher targets")
    parser.add_argument("--config", default="configs/distill_pune.yaml", help="Path to config")
    parser.add_argument("--max-patches", type=int, default=500, help="Max patches to cache")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for teacher inference")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    patches_dir = paths.get("patches_dir", "/mnt/pune/patches")
    cache_dir = paths.get("teacher_cache_dir", "/mnt/pune/teacher_cache")

    cache_teacher_targets(
        patches_dir=patches_dir,
        cache_dir=cache_dir,
        max_patches=args.max_patches,
        batch_size=args.batch_size,
    )
