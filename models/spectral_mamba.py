"""Wavelength-aware 1-D spectral sequence mixer."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class SpectralMamba(nn.Module):
    """Mix band tokens at every pixel while retaining all input bands."""

    def __init__(
        self,
        in_channels: int,
        d_model: int = 32,
        out_channels: int | None = None,
        wavelengths_nm: Sequence[float] | None = None,
        gsd_m: Sequence[float] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or d_model
        wavelengths = list(wavelengths_nm or range(in_channels))
        gsds = list(gsd_m or [1.0] * in_channels)
        if len(wavelengths) != in_channels or len(gsds) != in_channels:
            raise ValueError("Wavelength and GSD lists must match in_channels")
        self.value_embed = nn.Linear(1, d_model)
        self.register_buffer("wavelengths", torch.tensor(wavelengths, dtype=torch.float32), persistent=False)
        self.register_buffer("gsds", torch.tensor(gsds, dtype=torch.float32), persistent=False)
        self.position = nn.Linear(2, d_model)
        self.mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1),
        )
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.out = nn.Conv2d(d_model, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.wavelengths.numel():
            raise ValueError("SpectralMamba expects [B, in_channels, H, W]")
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(-1, c, 1)
        pos = torch.stack(
            [
                (self.wavelengths - self.wavelengths.mean()) / self.wavelengths.std().clamp_min(1e-6),
                (self.gsds - self.gsds.mean()) / self.gsds.std().clamp_min(1e-6),
            ],
            dim=-1,
        )
        tokens = self.value_embed(tokens) + self.position(pos).unsqueeze(0)
        mixed = self.mixer(tokens.transpose(1, 2)).transpose(1, 2)
        mixed = mixed * torch.sigmoid(self.gate(mixed))
        mixed = self.dropout(mixed).mean(dim=1)
        return self.out(mixed.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous())

