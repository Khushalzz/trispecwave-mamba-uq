"""Spatial state-space-style blocks.

If `mamba-ssm` is installed this module can be extended with a CUDA selective
scan implementation. The default path is a stable local state-space proxy:
depthwise directional mixing plus a recurrent-style gated residual. It keeps
the repository runnable on CPU and on ordinary PyTorch CUDA builds.
"""

from __future__ import annotations

import torch
from torch import nn

from .naf_blocks import LayerNorm2d


class SpatialMambaBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.horizontal = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels)
        self.vertical = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels)
        self.mix = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        state = self.horizontal(u) + self.vertical(u)
        state = self.mix(torch.tanh(state))
        return residual + self.scale * self.dropout(state * torch.sigmoid(gate))


class SpatialMamba(nn.Module):
    def __init__(self, channels: int, depth: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[SpatialMambaBlock(channels, dropout) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

