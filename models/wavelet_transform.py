"""Small differentiable Haar DWT/IDWT implementation."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """One-level orthogonal Haar decomposition.

    The high-frequency tensor is concatenated as `[LH, HL, HH]` along the
    channel axis. Odd spatial dimensions are padded and cropped by `inverse`.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError("DWT expects [B, C, H, W]")
        h, w = x.shape[-2:]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        x00, x01 = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
        x10, x11 = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        low = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return low, torch.cat([lh, hl, hh], dim=1), (h, w)

    @staticmethod
    def inverse(low: torch.Tensor, high: torch.Tensor, original_size: tuple[int, int] | None = None) -> torch.Tensor:
        if high.shape[1] != low.shape[1] * 3:
            raise ValueError("High-frequency tensor must contain three subbands per low channel")
        lh, hl, hh = high.chunk(3, dim=1)
        b, c, h, w = low.shape
        out = low.new_empty((b, c, h * 2, w * 2))
        out[..., 0::2, 0::2] = (low + lh + hl + hh) * 0.5
        out[..., 0::2, 1::2] = (low - lh + hl - hh) * 0.5
        out[..., 1::2, 0::2] = (low + lh - hl - hh) * 0.5
        out[..., 1::2, 1::2] = (low - lh - hl + hh) * 0.5
        if original_size is not None:
            out = out[..., : original_size[0], : original_size[1]]
        return out

