"""Spatial reconstruction metrics with dependency-light implementations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None, eps: float = 1e-12) -> torch.Tensor:
    mse = (pred - target).square().flatten(1).mean(dim=1)
    if data_range is None:
        data_range = float((target.amax() - target.amin()).detach().clamp_min(1e-6).item())
    return 10 * torch.log10(torch.as_tensor(data_range, device=pred.device).square() / mse.clamp_min(eps)).mean()


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None, window: int = 7) -> torch.Tensor:
    if data_range is None:
        data_range = float((target.amax() - target.amin()).detach().clamp_min(1e-6).item())
    pad = window // 2
    mu_x = F.avg_pool2d(pred, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, window, stride=1, padding=pad) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window, stride=1, padding=pad) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, window, stride=1, padding=pad) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
    return value.mean()


def lpips_proxy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A lightweight LPIPS-like feature distance when the LPIPS package is absent."""

    def features(x: torch.Tensor) -> list[torch.Tensor]:
        return [x, F.avg_pool2d(x, 2), F.avg_pool2d(x, 4)]

    return torch.stack([(a - b).abs().mean() for a, b in zip(features(pred), features(target))]).mean()

