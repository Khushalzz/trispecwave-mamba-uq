"""Spectral-safe paired augmentations."""

from __future__ import annotations

import random

import torch


class SpectralSafeAugmentation:
    """Apply the same spatial transform to every band and target."""

    def __init__(self, probability: float = 0.5, seed: int | None = None) -> None:
        self.probability = probability
        self.rng = random.Random(seed)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        if self.rng.random() > self.probability:
            return x, y, mask
        k = self.rng.randrange(4)
        x = torch.rot90(x, k, dims=(-2, -1))
        y = torch.rot90(y, k, dims=(-2, -1))
        mask = torch.rot90(mask, k, dims=(-2, -1))
        if self.rng.random() < 0.5:
            x, y, mask = x.flip(-1), y.flip(-1), mask.flip(-1)
        if self.rng.random() < 0.5:
            x, y, mask = x.flip(-2), y.flip(-2), mask.flip(-2)
        return x.contiguous(), y.contiguous(), mask.contiguous()

