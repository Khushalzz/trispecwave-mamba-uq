"""Sliding-window scene inference with smooth blending."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


class PatchTiler:
    def __init__(self, patch_size: int = 128, overlap: int = 16, scale: int = 4) -> None:
        if overlap >= patch_size:
            raise ValueError("overlap must be smaller than patch_size")
        self.patch_size = patch_size
        self.overlap = overlap
        self.scale = scale

    def _positions(self, length: int) -> list[int]:
        stride = self.patch_size - self.overlap
        if length <= self.patch_size:
            return [0]
        positions = list(range(0, max(1, length - self.patch_size + 1), stride))
        last = length - self.patch_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        one = torch.hann_window(self.patch_size, periodic=False, device=device, dtype=dtype)
        window = one[:, None] * one[None, :]
        return window.clamp_min(1e-3)

    @staticmethod
    def _prediction(value):
        if isinstance(value, dict):
            return value["mean"], value.get("logvar")
        if isinstance(value, tuple):
            return value[0], value[1] if len(value) > 1 else None
        return value, None

    @torch.no_grad()
    def predict(self, model: Callable, image: torch.Tensor, device: torch.device | str | None = None):
        """Return blended high-resolution mean and optional log-variance."""

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("image must be [C,H,W] or [B,C,H,W]")
        if image.shape[0] != 1:
            raise ValueError("PatchTiler currently processes one scene at a time")
        device = torch.device(device or next(model.parameters()).device)
        image = image.to(device)
        _, channels, height, width = image.shape
        out_h, out_w = height * self.scale, width * self.scale
        mean_acc = torch.zeros((1, channels, out_h, out_w), device=device, dtype=image.dtype)
        weight_acc = torch.zeros((1, 1, out_h, out_w), device=device, dtype=image.dtype)
        logvar_acc = torch.zeros_like(mean_acc)
        logvar_weight = torch.zeros_like(weight_acc)
        window = self._window(device, image.dtype)
        for top in self._positions(height):
            for left in self._positions(width):
                bottom, right = min(top + self.patch_size, height), min(left + self.patch_size, width)
                patch = image[..., top:bottom, left:right]
                valid_h, valid_w = patch.shape[-2:]
                if valid_h != self.patch_size or valid_w != self.patch_size:
                    patch = F.pad(patch, (0, self.patch_size - valid_w, 0, self.patch_size - valid_h), mode="replicate")
                mean, logvar = self._prediction(model(patch))
                mean = mean[..., : valid_h * self.scale, : valid_w * self.scale]
                high_window = F.interpolate(window[None, None], size=mean.shape[-2:], mode="bilinear", align_corners=False)
                y0, x0 = top * self.scale, left * self.scale
                y1, x1 = y0 + mean.shape[-2], x0 + mean.shape[-1]
                mean_acc[..., y0:y1, x0:x1] += mean * high_window
                weight_acc[..., y0:y1, x0:x1] += high_window
                if logvar is not None:
                    logvar = logvar[..., : valid_h * self.scale, : valid_w * self.scale]
                    logvar_acc[..., y0:y1, x0:x1] += logvar * high_window
                    logvar_weight[..., y0:y1, x0:x1] += high_window
        mean = mean_acc / weight_acc.clamp_min(1e-6)
        logvar = logvar_acc / logvar_weight.clamp_min(1e-6) if logvar_weight.max() > 0 else None
        return mean.squeeze(0), None if logvar is None else logvar.squeeze(0)

