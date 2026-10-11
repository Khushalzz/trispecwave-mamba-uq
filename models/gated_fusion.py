"""Cross-resolution learned gating."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CrossResolutionGatedFusion(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gates = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 1),
        )
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, native_10m: torch.Tensor, native_20m: torch.Tensor, native_60m: torch.Tensor) -> torch.Tensor:
        size = native_10m.shape[-2:]
        mid = F.interpolate(native_20m, size=size, mode="bilinear", align_corners=False)
        coarse = F.interpolate(native_60m, size=size, mode="bilinear", align_corners=False)
        weights = self.gates(torch.cat([native_10m, mid, coarse], dim=1)).softmax(dim=1)
        fused = native_10m + weights[:, :1] * mid + weights[:, 1:2] * coarse
        return fused + self.dropout(self.refine(fused))

