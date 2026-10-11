"""Prepare 64x64 Sentinel-2 L2A patches and generate manifest.csv.

Converts GeoTIFF reflectance to float32 [0, 1], filters clouds/shadows via SCL,
partitions splits strictly by acquisition date, and saves patch tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_pune")

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
# SCL values representing cloud, cloud shadow, cirrus, or invalid data
SCL_BAD_CLASSES = {0, 1, 3, 8, 9, 10, 11}


def prepare_pune_patches(
    raw_dir: str = "/mnt/pune/raw",
    patches_dir: str = "/mnt/pune/patches",
    patch_size: int = 64,
    stride: int = 32,
    max_cloud_fraction: float = 0.02,
    max_patches: int = 2000,
    clean_existing: bool = True,
) -> str:
    """Extract patches from downloaded scenes and write manifest.csv."""
    raw_path = Path(raw_dir)
    patches_path = Path(patches_dir)
    patches_path.mkdir(parents=True, exist_ok=True)
    manifest_path = patches_path / "manifest.csv"

    if clean_existing and patches_path.exists():
        logger.info(f"Purging existing patch files in {patches_path} to reclaim storage...")
        for old_pt in patches_path.glob("*.pt"):
            try:
                old_pt.unlink()
            except Exception:
                pass
        old_manifest = patches_path / "manifest.csv"
        if old_manifest.exists():
            old_manifest.unlink()

    manifest_json_path = raw_path / "scenes_manifest.json"
    if not manifest_json_path.exists():
        raise FileNotFoundError(f"Scenes manifest not found at {manifest_json_path}. Run download first.")

    with open(manifest_json_path) as f:
        scenes = json.load(f)

    if not scenes:
        raise ValueError("No scenes found in scenes manifest.")

    # Sort scenes chronologically
    scenes = sorted(scenes, key=lambda s: s.get("acquisition_date", ""))
    num_scenes = len(scenes)

    # Assign split strictly by acquisition date:
    # 3 train, 1 val, 1 test (if 5 scenes) or train/val if fewer
    scene_splits: dict[str, str] = {}
    if num_scenes >= 5:
        for s in scenes[:3]:
            scene_splits[s["scene_id"]] = "train"
        scene_splits[scenes[3]["scene_id"]] = "val"
        for s in scenes[4:]:
            scene_splits[s["scene_id"]] = "test"
    elif num_scenes >= 3:
        for s in scenes[:-2]:
            scene_splits[s["scene_id"]] = "train"
        scene_splits[scenes[-2]["scene_id"]] = "val"
        scene_splits[scenes[-1]["scene_id"]] = "test"
    else:
        scene_splits[scenes[0]["scene_id"]] = "train"
        scene_splits[scenes[-1]["scene_id"]] = "val"

    logger.info("Scene split assignments by acquisition date:")
    for sid, split in scene_splits.items():
        logger.info(f"  {sid}: {split}")

    # Allocate quota per split: 70% train, 15% val, 15% test
    train_scenes = [s for s in scenes if scene_splits[s["scene_id"]] == "train"]
    val_scenes = [s for s in scenes if scene_splits[s["scene_id"]] == "val"]
    test_scenes = [s for s in scenes if scene_splits[s["scene_id"]] == "test"]

    train_quota = int(max_patches * 0.70)
    val_quota = int(max_patches * 0.15) if val_scenes else 0
    test_quota = max_patches - train_quota - val_quota if test_scenes else 0

    scene_quota = {}
    base_train = train_quota // max(1, len(train_scenes))
    rem_train = train_quota % max(1, len(train_scenes))
    for idx, s in enumerate(train_scenes):
        scene_quota[s["scene_id"]] = base_train + (1 if idx < rem_train else 0)

    base_val = val_quota // max(1, len(val_scenes)) if val_scenes else 0
    rem_val = val_quota % max(1, len(val_scenes)) if val_scenes else 0
    for idx, s in enumerate(val_scenes):
        scene_quota[s["scene_id"]] = base_val + (1 if idx < rem_val else 0)

    base_test = test_quota // max(1, len(test_scenes)) if test_scenes else 0
    rem_test = test_quota % max(1, len(test_scenes)) if test_scenes else 0
    for idx, s in enumerate(test_scenes):
        scene_quota[s["scene_id"]] = base_test + (1 if idx < rem_test else 0)

    logger.info(f"Target quotas per scene: {scene_quota}")

    manifest_rows = []

    for scene in scenes:
        scene_id = scene["scene_id"]
        split = scene_splits[scene_id]
        acq_date = scene.get("acquisition_date", "")
        scene_dir = raw_path / scene_id
        target_scene_patches = scene_quota.get(scene_id, 100)
        scene_patches_extracted = 0

        # Load SCL cloud mask
        scl_path = scene_dir / "SCL.tif"
        if not scl_path.exists():
            logger.warning(f"SCL.tif missing for scene {scene_id}, skipping")
            continue

        with rasterio.open(scl_path) as scl_src:
            scl_data = scl_src.read(1)
            crs_str = str(scl_src.crs)
            scene_transform = scl_src.transform

        # Load all 12 bands in exact specified order
        band_arrays = []
        for b in BAND_NAMES:
            b_path = scene_dir / f"{b}.tif"
            with rasterio.open(b_path) as b_src:
                arr = b_src.read(1).astype(np.float32)
                # Convert Sentinel-2 L2A integer reflectance (0-10000) to float32 (0.0-1.0)
                arr = np.clip(arr / 10000.0, 0.0, 1.0)
                band_arrays.append(arr)

        scene_12band = np.stack(band_arrays, axis=0)  # [12, H, W]
        _, h, w = scene_12band.shape

        # Slide window across scene
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                if scene_patches_extracted >= target_scene_patches:
                    break

                # Extract SCL window and check cloud fraction
                scl_patch = scl_data[y : y + patch_size, x : x + patch_size]
                bad_pixels = np.isin(scl_patch, list(SCL_BAD_CLASSES)).sum()
                cloud_fraction = float(bad_pixels) / float(patch_size * patch_size)

                if cloud_fraction > max_cloud_fraction:
                    continue

                # Extract 12-band patch (must use .copy() to decouple from the 165MB scene parent array!)
                patch_12band = scene_12band[:, y : y + patch_size, x : x + patch_size].copy()
                if np.isnan(patch_12band).any() or (patch_12band.max() == 0.0):
                    continue

                # Compute geographic coordinates for patch center
                pixel_x = x + patch_size / 2.0
                pixel_y = y + patch_size / 2.0
                geo_x, geo_y = rasterio.transform.xy(scene_transform, pixel_y, pixel_x)

                patch_id = f"{scene_id}_y{y:04d}_x{x:04d}"
                out_patch_file = patches_path / f"{patch_id}.pt"

                patch_tensor = torch.from_numpy(patch_12band).float().contiguous()
                patch_metadata = {
                    "patch_id": patch_id,
                    "scene_id": scene_id,
                    "acquisition_date": acq_date,
                    "latitude": float(geo_y),
                    "longitude": float(geo_x),
                    "crs": crs_str,
                    "transform": [float(v) for v in scene_transform],
                    "cloud_fraction": cloud_fraction,
                    "split": split,
                    "patch_size": patch_size,
                    "out_channels": 12,
                }

                torch.save({"input": patch_tensor, "metadata": patch_metadata}, out_patch_file)

                manifest_rows.append(
                    {
                        "input_path": str(out_patch_file),
                        "scene_id": scene_id,
                        "acquisition_date": acq_date,
                        "latitude": float(geo_y),
                        "longitude": float(geo_x),
                        "crs": crs_str,
                        "transform": " ".join(f"{v:.6f}" for v in scene_transform),
                        "cloud_fraction": f"{cloud_fraction:.4f}",
                        "split": split,
                        "patch_id": patch_id,
                    }
                )
                scene_patches_extracted += 1

            if scene_patches_extracted >= target_scene_patches:
                break

    # Write manifest.csv
    fieldnames = [
        "input_path",
        "scene_id",
        "acquisition_date",
        "latitude",
        "longitude",
        "crs",
        "transform",
        "cloud_fraction",
        "split",
        "patch_id",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(f"Prepared {len(manifest_rows)} patches at {patches_dir}. Manifest saved to {manifest_path}.")
    counts = {}
    for r in manifest_rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    logger.info(f"Split distribution: {counts}")
    return str(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Pune 64x64 patches")
    parser.add_argument("--config", default="configs/distill_pune.yaml", help="Path to config")
    parser.add_argument("--max-patches", type=int, default=2000, help="Max patches to extract")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride")
    parser.add_argument("--no-clean", action="store_true", help="Do not purge existing patches before extracting")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    raw_dir = paths.get("raw_dir", "/mnt/pune/raw")
    patches_dir = paths.get("patches_dir", "/mnt/pune/patches")
    patch_size = cfg.get("patch_size", 64)

    prepare_pune_patches(
        raw_dir=raw_dir,
        patches_dir=patches_dir,
        patch_size=patch_size,
        stride=args.stride,
        max_patches=args.max_patches,
        clean_existing=not args.no_clean,
    )

