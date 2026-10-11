"""Sensor-consistency loss based on PSF/MTF degradation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def gaussian_kernel(size: int = 5, sigma: float = 1.2, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d[:, None] * kernel_1d[None, :]


class MTFPSFLoss(nn.Module):
    """Blur and downsample a high-resolution prediction to sensor resolution."""

    def __init__(self, scale: int = 4, kernel_size: int = 5, sigma: float = 1.2) -> None:
        super().__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma = sigma

    def degrade(self, high_res: torch.Tensor, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        b, c, _, _ = high_res.shape
        kernel = gaussian_kernel(self.kernel_size, self.sigma, high_res.device, high_res.dtype)
        kernel = kernel.expand(c, 1, -1, -1)
        blurred = F.conv2d(high_res, kernel, padding=self.kernel_size // 2, groups=c)
        if output_size is None:
            output_size = (max(1, high_res.shape[-2] // self.scale), max(1, high_res.shape[-1] // self.scale))
        return F.interpolate(blurred, size=output_size, mode="area")

    def forward(self, high_res: torch.Tensor, native_target: torch.Tensor) -> torch.Tensor:
        degraded = self.degrade(high_res, native_target.shape[-2:])
        return F.l1_loss(degraded, native_target)

