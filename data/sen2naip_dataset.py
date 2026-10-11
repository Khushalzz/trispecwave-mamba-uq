"""SEN2NAIP paired 10 m Sentinel-2 / high-resolution loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .base_raster_dataset import BaseRasterDataset, load_manifest


class Sen2NAIPDataset(BaseRasterDataset):
    """Read Sentinel-2 to NAIP pairs from a manifest."""

    def __init__(
        self,
        manifest: str | Path | Sequence[Mapping[str, Any]],
        input_root: str | Path | None = None,
        target_root: str | Path | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> None:
        if split is not None:
            manifest = [row for row in load_manifest(manifest) if row.get("split", split) == split]
        super().__init__(manifest, input_root=input_root, target_root=target_root, **kwargs)
