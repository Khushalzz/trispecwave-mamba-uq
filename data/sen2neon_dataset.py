"""SEN2NEON/AVIRIS-NG multispectral pair loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .base_raster_dataset import BaseRasterDataset


class Sen2NEONDataset(BaseRasterDataset):
    """Loader with optional explicit band selection for 12-band supervision."""

    def __init__(
        self,
        manifest: str | Path | Sequence[Mapping[str, Any]],
        input_root: str | Path | None = None,
        target_root: str | Path | None = None,
        band_indices: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(manifest, input_root=input_root, target_root=target_root, **kwargs)
        self.band_indices = list(band_indices) if band_indices is not None else None

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        if self.band_indices is not None:
            sample["input"] = sample["input"][self.band_indices]
            sample["target"] = sample["target"][self.band_indices]
        return sample

