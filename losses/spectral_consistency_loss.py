"""Spectral-angle and vegetation/water index consistency losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return mean spectral angle in radians."""

    dot = (pred * target).sum(dim=1)
    pred_norm = pred.square().sum(dim=1).sqrt().clamp_min(eps)
    target_norm = target.square().sum(dim=1).sqrt().clamp_min(eps)
    cosine = (dot / (pred_norm * target_norm)).clamp(-1 + eps, 1 - eps)
    return torch.acos(cosine).mean()


def _index(image: torch.Tensor, numerator_band: int, denominator_band: int, eps: float = 1e-6) -> torch.Tensor:
    numerator = image[:, numerator_band : numerator_band + 1]
    denominator = image[:, denominator_band : denominator_band + 1]
    return (numerator - denominator) / (numerator + denominator).abs().clamp_min(eps)


class SpectralConsistencyLoss(nn.Module):
    """SAM plus NDVI/NDWI preservation for Sentinel-2 band ordering."""

    def __init__(self, sam_weight: float = 1.0, index_weight: float = 0.25) -> None:
        super().__init__()
        self.sam_weight = sam_weight
        self.index_weight = index_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
        loss = self.sam_weight * spectral_angle_mapper(pred, target)
        if pred.shape[1] >= 8:
            pred_ndvi = _index(pred, 7, 3)
            target_ndvi = _index(target, 7, 3)
            loss = loss + self.index_weight * F.l1_loss(pred_ndvi, target_ndvi)
        if pred.shape[1] >= 3:
            pred_ndwi = _index(pred, 2, 7 if pred.shape[1] > 7 else 1)
            target_ndwi = _index(target, 2, 7 if target.shape[1] > 7 else 1)
            loss = loss + self.index_weight * F.l1_loss(pred_ndwi, target_ndwi)
        return loss

