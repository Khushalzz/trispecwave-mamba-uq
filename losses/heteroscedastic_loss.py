"""Heteroscedastic aleatoric negative log-likelihood losses."""

from __future__ import annotations

import torch


def _expand_mask(mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = mask.to(device=target.device, dtype=target.dtype)
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != target.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=target.shape[-2:], mode="nearest")
    return mask.expand_as(target)


def heteroscedastic_l1_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    clamp: tuple[float, float] = (-10.0, 5.0),
) -> torch.Tensor:
    """Compute 0.5 * exp(-s) * |y - mu| + 0.5 * s."""

    if mean.shape != target.shape or logvar.shape != target.shape:
        raise ValueError("mean, logvar, and target must have the same shape")
    logvar = logvar.clamp(*clamp)
    loss = 0.5 * (torch.exp(-logvar) * (target - mean).abs() + logvar)
    expanded = _expand_mask(mask, target)
    if expanded is not None:
        loss = loss * expanded
        return loss.sum() / expanded.sum().clamp_min(1.0)
    return loss.mean()

