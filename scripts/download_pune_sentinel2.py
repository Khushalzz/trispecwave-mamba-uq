"""Search and download Sentinel-2 L2A scenes around Pune, Maharashtra, India.

Queries Copernicus Data Space STAC or Planetary Computer STAC for low-cloud scenes,
extracts the 12 required bands (excluding B10) plus SCL for the specified AOI,
and stores windowed GeoTIFFs and manifest on the persistent Modal volume.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_pune")

# Exact Sentinel-2 L2A 12-band order (excluding B10)
BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

# Mapping from our 12-band names to STAC asset keys
STAC_ASSET_KEYS = {
    "B1": "B01",
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B5": "B05",
    "B6": "B06",
    "B7": "B07",
    "B8": "B08",
    "B8A": "B8A",
    "B9": "B09",
    "B11": "B11",
    "B12": "B12",
    "SCL": "SCL",
}

# Native spatial resolution for each band in meters
NATIVE_GSD_M = {
    "B1": 60.0,
    "B2": 10.0,
    "B3": 10.0,
    "B4": 10.0,
    "B5": 20.0,
    "B6": 20.0,
    "B7": 20.0,
    "B8": 10.0,
    "B8A": 20.0,
    "B9": 60.0,
    "B11": 20.0,
    "B12": 20.0,
    "SCL": 20.0,
}


def compute_bbox(lat: float, lon: float, radius_km: float = 10.0) -> list[float]:
    """Compute bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84."""
    # Approx 111 km per degree latitude
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    return [
        round(lon - lon_delta, 4),
        round(lat - lat_delta, 4),
        round(lon + lon_delta, 4),
        round(lat + lat_delta, 4),
    ]


def search_copernicus_stac(bbox: list[float], max_cloud: float = 10.0) -> list[dict[str, Any]]:
    """Attempt search on Copernicus Data Space STAC."""
    url = "https://stac.dataspace.copernicus.eu/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "query": {"cloudCover": {"lt": max_cloud}},
        "datetime": "2024-01-01T00:00:00Z/2024-05-30T23:59:59Z",
        "limit": 20,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            features = resp.json().get("features", [])
            if features:
                logger.info(f"Copernicus STAC returned {len(features)} items")
                return features
    except Exception as exc:
        logger.warning(f"Copernicus STAC query failed: {exc}")
    return []


def search_planetary_computer_stac(bbox: list[float], max_cloud: float = 5.0) -> list[dict[str, Any]]:
    """Query Microsoft Planetary Computer STAC and select 3-5 distinct monthly scenes."""
    import planetary_computer

    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "datetime": "2024-01-01T00:00:00Z/2024-05-30T23:59:59Z",
        "limit": 50,
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    logger.info(f"Planetary Computer found {len(features)} low-cloud scenes over Pune.")

    # Group by month and pick the lowest-cloud scene per month
    by_month: dict[str, dict[str, Any]] = {}
    for f in features:
        props = f.get("properties", {})
        dt = props.get("datetime", "")[:7]  # YYYY-MM
        cloud = props.get("eo:cloud_cover", 100.0)
        if dt not in by_month or cloud < by_month[dt]["properties"].get("eo:cloud_cover", 100.0):
            by_month[dt] = f

    # Sort chronologically
    selected = [by_month[k] for k in sorted(by_month.keys())]
    # Sign assets
    signed_items = []
    for item in selected:
        signed_items.append(planetary_computer.sign(item))
    return signed_items


def download_and_crop_scene(
    item: dict[str, Any],
    output_scene_dir: Path,
    bbox: list[float],
) -> dict[str, Any]:
    """Download windowed GeoTIFFs for all 12 bands and SCL onto a common 10m grid."""
    from rasterio.warp import transform_bounds

    output_scene_dir.mkdir(parents=True, exist_ok=True)
    scene_id = item["id"]
    props = item.get("properties", {})
    acq_date = props.get("datetime", "")
    cloud_cover = props.get("eo:cloud_cover", 0.0)

    # 1. Open reference 10m band (B02) to establish master 10m grid transform and shape
    b2_href = item["assets"]["B02"]["href"]
    with rasterio.open(b2_href) as ref_src:
        ref_crs = ref_src.crs
        # Project WGS84 bbox into scene CRS (UTM 43N)
        left, bottom, right, top = transform_bounds("EPSG:4326", ref_crs, *bbox)
        ref_window = from_bounds(left, bottom, right, top, ref_src.transform)
        # Snap window to integer pixels
        ref_window = ref_window.round_offsets().round_lengths()
        ref_transform = rasterio.windows.transform(ref_window, ref_src.transform)
        out_height = int(ref_window.height)
        out_width = int(ref_window.width)

        # Ensure height and width are multiples of 64 for clean patching
        out_height = (out_height // 64) * 64
        out_width = (out_width // 64) * 64

    logger.info(
        f"Processing scene {scene_id} ({acq_date}): grid {out_width}x{out_height} (10m pixels) CRS={ref_crs}"
    )

    saved_band_files: dict[str, str] = {}

    # 2. Extract each required band and SCL
    for band_name in BAND_NAMES + ["SCL"]:
        out_path = output_scene_dir / f"{band_name}.tif"
        if out_path.exists() and out_path.stat().st_size > 1000:
            saved_band_files[band_name] = str(out_path)
            continue

        asset_key = STAC_ASSET_KEYS[band_name]
        if asset_key not in item["assets"]:
            raise KeyError(f"Asset key {asset_key} for band {band_name} not found in scene {scene_id}")

        asset_href = item["assets"][asset_key]["href"]
        with rasterio.open(asset_href) as src:
            # Recompute window in this band's coordinates
            band_window = from_bounds(left, bottom, right, top, src.transform)
            # Read and resample into the reference 10m grid dimensions
            resampling = Resampling.nearest if band_name == "SCL" else Resampling.bilinear
            data = src.read(
                1,
                window=band_window,
                out_shape=(out_height, out_width),
                resampling=resampling,
            )

            # Write cropped 10m GeoTIFF
            out_meta = {
                "driver": "GTiff",
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "width": out_width,
                "height": out_height,
                "count": 1,
                "crs": ref_crs,
                "transform": ref_transform,
                "compress": "lzw",
            }
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(data, 1)

        saved_band_files[band_name] = str(out_path)

    metadata = {
        "scene_id": scene_id,
        "acquisition_date": acq_date,
        "cloud_cover": cloud_cover,
        "crs": str(ref_crs),
        "transform": [float(x) for x in ref_transform],
        "width": out_width,
        "height": out_height,
        "bands": saved_band_files,
    }
    with open(output_scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def download_pune_data(
    raw_dir: str = "/mnt/pune/raw",
    lat: float = 18.4388,
    lon: float = 73.7728,
    radius_km: float = 10.0,
    max_scenes: int = 5,
) -> list[dict[str, Any]]:
    """Main pipeline for searching and downloading Pune Sentinel-2 L2A scenes."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_path / "scenes_manifest.json"

    # Check if existing scenes are already present
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = json.load(f)
        if len(existing) >= 3:
            all_valid = True
            for sc in existing:
                sc_dir = raw_path / sc["scene_id"]
                if not sc_dir.exists() or len(list(sc_dir.glob("*.tif"))) < 13:
                    all_valid = False
                    break
            if all_valid:
                logger.info(f"Existing {len(existing)} scenes found on volume. Skipping download.")
                return existing

    bbox = compute_bbox(lat, lon, radius_km)
    logger.info(f"Pune AOI Bounding Box: {bbox}")

    # Try Copernicus first, fallback to Planetary Computer
    scenes = search_copernicus_stac(bbox, max_cloud=10.0)
    if not scenes:
        logger.info("Copernicus STAC returned no usable scenes; using Microsoft Planetary Computer STAC.")
        scenes = search_planetary_computer_stac(bbox, max_cloud=5.0)

    if not scenes:
        raise RuntimeError("No low-cloud Sentinel-2 scenes found for Pune AOI.")

    scenes = scenes[:max_scenes]
    logger.info(f"Downloading and cropping {len(scenes)} scenes to {raw_dir}...")

    manifest = []
    for item in scenes:
        scene_dir = raw_path / item["id"]
        meta = download_and_crop_scene(item, scene_dir, bbox)
        manifest.append(meta)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Completed downloading {len(manifest)} scenes to {raw_dir}. Manifest saved.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Pune Sentinel-2 L2A data")
    parser.add_argument("--config", default="configs/distill_pune.yaml", help="Config file path")
    parser.add_argument("--raw-dir", default="/mnt/pune/raw", help="Target raw directory")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    aoi = cfg.get("aoi", {})
    lat = aoi.get("latitude", 18.4388)
    lon = aoi.get("longitude", 73.7728)
    radius = aoi.get("radius_km", 10.0)
    raw_dir = cfg.get("paths", {}).get("raw_dir", args.raw_dir)

    download_pune_data(raw_dir=raw_dir, lat=lat, lon=lon, radius_km=radius)
