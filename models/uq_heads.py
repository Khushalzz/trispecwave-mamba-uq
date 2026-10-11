"""Mean and heteroscedastic uncertainty heads."""

from __future__ import annotations

import torch
from torch import nn


class UQHeads(nn.Module):
    def __init__(self, channels: int, out_channels: int, min_logvar: float = -10.0, max_logvar: float = 5.0) -> None:
        super().__init__()
        self.mean = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.logvar = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(x)
        logvar = self.logvar(x).clamp(self.min_logvar, self.max_logvar)
        return mean, logvar

