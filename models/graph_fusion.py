"""Lightweight inter-band graph attention without a hard PyG dependency."""

from __future__ import annotations

import torch
from torch import nn


class SpectralGraphMixer(nn.Module):
    """Learned graph message passing over native-resolution feature nodes."""

    def __init__(self, channels: int, num_nodes: int = 3, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        self.heads = max(1, heads)
        self.edge_logits = nn.Parameter(torch.zeros(self.heads, num_nodes, num_nodes))
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, nodes: torch.Tensor, return_nodes: bool = False):
        if nodes.ndim != 5 or nodes.shape[1] != self.num_nodes:
            raise ValueError(f"Expected [B, {self.num_nodes}, C, H, W] graph nodes")
        adjacency = self.edge_logits.softmax(dim=-1).mean(dim=0)
        mixed = torch.einsum("nm,bmchw->bnchw", adjacency, nodes)
        q = self.q(nodes.flatten(0, 1)).view_as(nodes)
        k = self.k(nodes.flatten(0, 1)).view_as(nodes)
        v = self.v(nodes.flatten(0, 1)).view_as(nodes)
        affinity = (q * k).mean(dim=2, keepdim=True)
        affinity = affinity.softmax(dim=1)
        mixed = mixed + affinity * v
        fused = self.out(self.norm(mixed.mean(dim=1)))
        fused = fused + nodes.mean(dim=1)
        fused = self.dropout(fused)
        return (fused, mixed) if return_nodes else fused

