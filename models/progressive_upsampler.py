"""Progressive 2x + 2x feature upsampler."""

from __future__ import annotations

import torch
from torch import nn

from .naf_blocks import NAFBlock


class UpsampleStage(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.refine = NAFBlock(channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.shuffle(self.expand(x)))


class ProgressiveUpsampler(nn.Module):
    def __init__(self, channels: int, stages: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be >= 1")
        self.stages = nn.ModuleList([UpsampleStage(channels, dropout) for _ in range(stages)])

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return (x, outputs) if return_intermediates else x

