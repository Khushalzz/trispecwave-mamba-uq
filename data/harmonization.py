"""Radiometric and geometric harmonization utilities."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def radiometric_normalize(source: torch.Tensor, reference: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match per-band mean and standard deviation of source to reference."""

    if source.ndim != reference.ndim or source.shape[0] != reference.shape[0]:
        raise ValueError("Source and reference must have matching batch/channel dimensions")
    dims = tuple(range(2, source.ndim)) if source.ndim >= 4 else tuple(range(1, source.ndim))
    source_mean = source.mean(dim=dims, keepdim=True)
    source_std = source.std(dim=dims, keepdim=True).clamp_min(eps)
    reference_mean = reference.mean(dim=dims, keepdim=True)
    reference_std = reference.std(dim=dims, keepdim=True).clamp_min(eps)
    return (source - source_mean) / source_std * reference_std + reference_mean


def translate_tensor(image: torch.Tensor, shift_y: float, shift_x: float) -> torch.Tensor:
    """Sub-pixel translation using a differentiable affine sampler."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    n, _, h, w = image.shape
    theta = image.new_zeros((n, 2, 3))
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2 * shift_x / max(w - 1, 1)
    theta[:, 1, 2] = -2 * shift_y / max(h - 1, 1)
    grid = F.affine_grid(theta, image.shape, align_corners=True)
    result = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return result.squeeze(0) if squeeze else result


def estimate_translation(reference: torch.Tensor, moving: torch.Tensor) -> Tuple[float, float]:
    """Estimate integer translation by a small exhaustive correlation search."""

    ref = reference.float().mean(dim=0) if reference.ndim == 3 else reference.float().mean(dim=1)[0]
    mov = moving.float().mean(dim=0) if moving.ndim == 3 else moving.float().mean(dim=1)[0]
    best = (-float("inf"), 0, 0)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            shifted = torch.roll(mov, shifts=(dy, dx), dims=(-2, -1))
            score = float((ref * shifted).mean().item())
            if score > best[0]:
                best = (score, dy, dx)
    return float(best[1]), float(best[2])

