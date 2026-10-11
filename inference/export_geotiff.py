"""GeoTIFF/COG export while preserving spatial reference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:
    rasterio = None
    Affine = None


def _to_chw(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None]
    if array.ndim == 3 and array.shape[0] <= 32:
        return array
    if array.ndim == 3:
        return np.moveaxis(array, -1, 0)
    raise ValueError("Expected [C,H,W], [H,W,C], or [H,W]")


def export_geotiff(
    path: str | Path,
    prediction: np.ndarray | torch.Tensor,
    reference_profile: dict[str, Any],
    transform: Any | None = None,
    uncertainty: np.ndarray | torch.Tensor | None = None,
    scale: int = 4,
    nodata: float | None = None,
) -> Path:
    """Write prediction and optional uncertainty bands to a COG-compatible TIFF."""

    if rasterio is None:
        raise ImportError("Rasterio is required for GeoTIFF export")
    output = Path(path)
    pred = _to_chw(prediction).astype(np.float32)
    unc = None if uncertainty is None else _to_chw(uncertainty).astype(np.float32)
    data = pred if unc is None else np.concatenate([pred, unc], axis=0)
    profile = dict(reference_profile)
    profile.update(
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32",
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    if nodata is not None:
        profile["nodata"] = nodata
    if transform is None:
        transform = profile.get("transform")
    if transform is not None and Affine is not None:
        profile["transform"] = transform * Affine.scale(1.0 / scale, 1.0 / scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dataset:
        dataset.write(data)
        try:
            dataset.build_overviews([2, 4, 8], rasterio.enums.Resampling.average)
            dataset.update_tags(ns="rio_overview", resampling="average")
        except Exception:
            pass
    return output

