"""Inference orchestration for raster files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from inference.export_geotiff import export_geotiff
from inference.mc_dropout_sampler import mc_dropout_predict
from inference.patch_tiler import PatchTiler
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from data.base_raster_dataset import read_raster


class InferencePipeline:
    def __init__(self, model: torch.nn.Module, device: str | torch.device = "cpu", tiler: PatchTiler | None = None) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tiler = tiler or PatchTiler()

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, model_kwargs: dict[str, Any] | None = None, device: str = "cpu"):
        model = TriSpecWaveMambaUQ(**(model_kwargs or {}))
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
        return cls(model, device=device)

    def run_tensor(self, image: torch.Tensor, samples: int = 1) -> dict[str, torch.Tensor | None]:
        image = image.to(self.device)
        if samples > 1:
            result = mc_dropout_predict(self.model, image.unsqueeze(0) if image.ndim == 3 else image, samples)
            return {"mean": result["mean"].squeeze(0), "logvar": result.get("logvar"), "uncertainty": result["total_variance"].squeeze(0)}
        mean, logvar = self.tiler.predict(self.model, image)
        return {"mean": mean, "logvar": logvar, "uncertainty": None if logvar is None else logvar.exp()}

    def run_file(self, input_path: str | Path, output_path: str | Path, uncertainty_path: str | Path | None = None, samples: int = 1):
        array, metadata = read_raster(input_path)
        result = self.run_tensor(torch.from_numpy(array), samples=samples)
        profile = metadata.profile or {"driver": "GTiff", "crs": metadata.crs, "transform": metadata.transform}
        export_geotiff(output_path, result["mean"], profile, transform=metadata.transform, uncertainty=result["uncertainty"], scale=self.tiler.scale)
        return result
