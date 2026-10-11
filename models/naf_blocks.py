"""Compact NAFNet-style restoration blocks."""

from __future__ import annotations

import torch
from torch import nn


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mlp(self.pool(x))


class NAFBlock(nn.Module):
    """NAF-inspired gated depthwise restoration block."""

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.gate = SimpleGate()
        self.sca = ChannelAttention(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        self.drop1 = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.norm2 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden * 2, 1),
            SimpleGate(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.drop2 = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw1(self.norm1(x))
        y = self.sca(self.gate(self.dw(y)))
        x = x + self.beta * self.drop1(self.pw2(y))
        x = x + self.gamma * self.drop2(self.ffn(self.norm2(x)))
        return x


class NAFStack(nn.Sequential):
    def __init__(self, channels: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__(*[NAFBlock(channels, dropout=dropout) for _ in range(depth)])
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self:
            if self.gradient_checkpointing and x.requires_grad:
                import torch.utils.checkpoint
                x = torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
            else:
                x = module(x)
        return x

