"""Spectral reconstruction metrics."""

from __future__ import annotations

import torch


def spectral_angle_mapper_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dot = (pred * target).sum(dim=1)
    denom = pred.square().sum(dim=1).sqrt().clamp_min(eps) * target.square().sum(dim=1).sqrt().clamp_min(eps)
    return torch.acos((dot / denom).clamp(-1 + eps, 1 - eps)).mean()


def ergas(pred: torch.Tensor, target: torch.Tensor, scale: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    rmse = (pred - target).square().flatten(2).mean(dim=2).sqrt()
    mean = target.flatten(2).mean(dim=2).abs().clamp_min(eps)
    return 100.0 / scale * torch.sqrt((rmse / mean).square().mean())


def band_correlation(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = pred.flatten(2) - pred.flatten(2).mean(dim=2, keepdim=True)
    t = target.flatten(2) - target.flatten(2).mean(dim=2, keepdim=True)
    corr = (p * t).sum(dim=2) / (p.square().sum(dim=2).sqrt() * t.square().sum(dim=2).sqrt()).clamp_min(eps)
    return corr.mean(dim=0)

