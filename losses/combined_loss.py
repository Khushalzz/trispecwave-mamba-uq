"""Master objective for TriSpecWave-Mamba-UQ training."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .heteroscedastic_loss import heteroscedastic_l1_loss
from .mtf_psf_loss import MTFPSFLoss
from .spectral_consistency_loss import SpectralConsistencyLoss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        mtf_scale: int = 4,
    ) -> None:
        super().__init__()
        self.weights = {
            "regression": 1.0,
            "uncertainty": 0.25,
            "spectral": 0.20,
            "mtf": 0.10,
            **dict(weights or {}),
        }
        self.spectral_loss = SpectralConsistencyLoss()
        self.mtf_loss = MTFPSFLoss(scale=mtf_scale)

    def forward(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
        native_input: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if target.shape != mean.shape:
            raise ValueError(f"Prediction {tuple(mean.shape)} does not match target {tuple(target.shape)}")
        regression = F.l1_loss(mean, target)
        uncertainty = heteroscedastic_l1_loss(mean, logvar, target, mask=mask)
        spectral = self.spectral_loss(mean, target)
        mtf = mean.new_zeros(())
        if native_input is not None:
            mtf = self.mtf_loss(mean, native_input)
        total = (
            self.weights["regression"] * regression
            + self.weights["uncertainty"] * uncertainty
            + self.weights["spectral"] * spectral
            + self.weights["mtf"] * mtf
        )
        parts = {"total": total.detach(), "regression": regression.detach(), "uncertainty": uncertainty.detach(), "spectral": spectral.detach(), "mtf": mtf.detach()}
        return total, parts

