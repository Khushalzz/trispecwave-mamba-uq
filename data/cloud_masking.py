"""Cloud and cloud-shadow filtering helpers."""

from __future__ import annotations

import numpy as np
import torch


def valid_cloud_fraction(cloud_mask: np.ndarray | torch.Tensor) -> float:
    """Return the fraction of pixels not marked as cloud/shadow."""

    mask = torch.as_tensor(cloud_mask).bool()
    return float((~mask).float().mean().item())


def apply_cloud_mask(image: np.ndarray | torch.Tensor, cloud_mask: np.ndarray | torch.Tensor, fill: float = 0.0):
    """Fill clouded pixels while retaining the original array/tensor type."""

    if isinstance(image, torch.Tensor):
        result = image.clone()
        mask = torch.as_tensor(cloud_mask, device=result.device).bool()
        while mask.ndim < result.ndim:
            mask = mask.unsqueeze(0)
        return result.masked_fill(mask, fill)
    result = np.array(image, copy=True)
    mask = np.asarray(cloud_mask).astype(bool)
    while mask.ndim < result.ndim:
        mask = np.expand_dims(mask, 0)
    result[mask] = fill
    return result


def accept_tile(cloud_mask: np.ndarray | torch.Tensor, max_cloud_fraction: float = 0.15) -> bool:
    """Whether a tile is suitable for training."""

    return (1.0 - valid_cloud_fraction(cloud_mask)) <= max_cloud_fraction

