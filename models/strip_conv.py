"""Directional high-frequency refinement blocks."""

from __future__ import annotations

import torch
from torch import nn


class StripConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.GroupNorm(1, channels)
        self.h = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, pad), groups=channels)
        self.v = nn.Conv2d(channels, channels, (kernel_size, 1), padding=(pad, 0), groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels, channels * 2, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.h(self.norm(x)) + self.v(self.norm(x))
        a, b = self.gate(y).chunk(2, dim=1)
        return x + self.dropout(self.pointwise(torch.tanh(a) * torch.sigmoid(b)))


class StripConvStack(nn.Sequential):
    def __init__(self, channels: int, depth: int = 2, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__(*[StripConvBlock(channels, kernel_size, dropout) for _ in range(depth)])
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self:
            if self.gradient_checkpointing and x.requires_grad:
                import torch.utils.checkpoint
                x = torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
            else:
                x = module(x)
        return x

