"""Base datasets for paired geospatial rasters.

The loader keeps geospatial metadata separate from tensors and returns CHW
float tensors. It accepts GeoTIFF paths as well as NumPy arrays, which makes
unit tests and synthetic experiments possible without GDAL/Rasterio.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError:
    rasterio = None


@dataclass
class RasterMetadata:
    """Small metadata container needed by geospatial export."""

    transform: Any = None
    crs: Any = None
    nodata: Any = None
    profile: dict[str, Any] | None = None


def _as_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D raster, got shape {array.shape}")
    if array.shape[0] <= 32 and array.shape[1] > 32 and array.shape[2] > 32:
        return array
    return np.moveaxis(array, -1, 0)


def read_raster(source: str | Path | np.ndarray | torch.Tensor) -> tuple[np.ndarray, RasterMetadata]:
    """Read a raster-like object and return a CHW NumPy array plus metadata."""

    if isinstance(source, torch.Tensor):
        return source.detach().cpu().numpy().astype(np.float32), RasterMetadata()
    if isinstance(source, np.ndarray):
        return _as_chw(source).astype(np.float32), RasterMetadata()
    if rasterio is None:
        raise ImportError("Rasterio is required to read file paths. Install requirements.txt.")
    with rasterio.open(str(source)) as dataset:
        array = dataset.read().astype(np.float32)
        metadata = RasterMetadata(
            transform=dataset.transform,
            crs=dataset.crs,
            nodata=dataset.nodata,
            profile=dataset.profile.copy(),
        )
    return array, metadata


def load_manifest(manifest: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Load CSV, JSON, JSONL, or an already materialized list of samples."""

    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, Path)):
        return [dict(row) for row in manifest]
    path = Path(manifest)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
        return [dict(row) for row in value]
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _valid_mask(array: np.ndarray, nodata: Any = None) -> np.ndarray:
    valid = np.isfinite(array).all(axis=0)
    if nodata is not None:
        valid &= (array != nodata).all(axis=0)
    return valid.astype(np.float32)


class BaseRasterDataset(Dataset):
    """Generic paired raster dataset.

    Each manifest row needs `input` and `target` keys. Optional `mask` values
    are interpreted as a single-band valid-pixel mask. A callable transform can
    jointly modify input, target, and mask while preserving band alignment.
    """

    def __init__(
        self,
        samples: str | Path | Sequence[Mapping[str, Any]],
        input_root: str | Path | None = None,
        target_root: str | Path | None = None,
        transform: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
        normalizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.samples = load_manifest(samples)
        self.input_root = Path(input_root) if input_root else None
        self.target_root = Path(target_root) if target_root else None
        self.transform = transform
        self.normalizer = normalizer
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _resolve(value: Any, root: Path | None) -> Any:
        if root is not None and isinstance(value, (str, Path)):
            path = Path(value)
            return path if path.is_absolute() else root / path
        return value

    @staticmethod
    def _tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.nan_to_num(array, nan=0.0).astype(np.float32))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        input_array, input_meta = read_raster(self._resolve(row["input"], self.input_root))
        target_source = row.get("target", row["input"])
        target_array, target_meta = read_raster(self._resolve(target_source, self.target_root))
        input_tensor = self._tensor(input_array)
        target_tensor = self._tensor(target_array)
        if "mask" in row and row["mask"] not in (None, ""):
            mask_array, _ = read_raster(self._resolve(row["mask"], self.input_root))
            mask = self._tensor(mask_array[:1]).clamp(0, 1)
        else:
            mask = torch.from_numpy(_valid_mask(input_array, input_meta.nodata)).unsqueeze(0)
            if target_tensor.shape[-2:] != input_tensor.shape[-2:]:
                mask = F.interpolate(mask.unsqueeze(0), size=target_tensor.shape[-2:], mode="nearest").squeeze(0)
        if self.normalizer is not None:
            input_tensor = self.normalizer(input_tensor)
        if self.transform is not None:
            input_tensor, target_tensor, mask = self.transform(input_tensor, target_tensor, mask)
        result: dict[str, Any] = {"input": input_tensor, "target": target_tensor, "mask": mask.float()}
        if self.return_metadata:
            result["metadata"] = {"input": input_meta, "target": target_meta, "row": row}
        return result

