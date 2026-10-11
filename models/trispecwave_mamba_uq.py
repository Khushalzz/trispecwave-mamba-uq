"""Top-level TriSpecWave-Mamba-UQ model."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .gated_fusion import CrossResolutionGatedFusion
from .graph_fusion import SpectralGraphMixer
from .naf_blocks import NAFStack
from .progressive_upsampler import ProgressiveUpsampler
from .spatial_mamba import SpatialMamba
from .spectral_mamba import SpectralMamba
from .strip_conv import StripConvStack
from .uq_heads import UQHeads
from .wavelet_transform import HaarDWT2D


DEFAULT_GROUPS = {
    "native_10m": [1, 2, 3, 7],
    "native_20m": [4, 5, 6, 8, 10, 11],
    "native_60m": [0, 9],
}


class TriSpecWaveMambaUQ(nn.Module):
    """Multi-stream spectral/spatial model with aleatoric uncertainty.

    Input is a 12-band tensor on a common 10 m feature grid. Native 20 m and
    60 m groups are represented by their resampled feature nodes; the original
    band values remain available through the residual mean path.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 12,
        feature_dim: int = 64,
        spectral_dim: int = 32,
        naf_depth: int = 4,
        spatial_depth: int = 2,
        graph_heads: int = 4,
        dropout: float = 0.1,
        groups: Mapping[str, Sequence[int]] | None = None,
        band_wavelengths_nm: Sequence[float] | None = None,
        band_gsd_m: Sequence[float] | None = None,
        wavelet_levels: int = 1,
        upsample_stages: int = 2,
    ) -> None:
        super().__init__()
        if wavelet_levels != 1:
            raise ValueError("The compact implementation currently supports one wavelet level")
        self.in_channels = in_channels
        self.out_channels = out_channels
        group_map = dict(groups or DEFAULT_GROUPS)
        self.group_names = ["native_10m", "native_20m", "native_60m"]
        self.group_indices = {name: [i for i in group_map.get(name, []) if i < in_channels] for name in self.group_names}
        if any(not indices for indices in self.group_indices.values()):
            raise ValueError("Each native band group must contain at least one input band")
        self.group_encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(len(indices), feature_dim, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                )
                for name, indices in self.group_indices.items()
            }
        )
        self.graph = SpectralGraphMixer(feature_dim, num_nodes=3, heads=graph_heads, dropout=dropout)
        self.spectral = SpectralMamba(
            in_channels=in_channels,
            d_model=spectral_dim,
            out_channels=feature_dim,
            wavelengths_nm=band_wavelengths_nm,
            gsd_m=band_gsd_m,
            dropout=dropout,
        )
        self.gated_fusion = CrossResolutionGatedFusion(feature_dim, dropout=dropout)
        self.pre_refine = NAFStack(feature_dim, depth=max(1, naf_depth // 2), dropout=dropout)
        self.dwt = HaarDWT2D()
        self.spatial = SpatialMamba(feature_dim, depth=spatial_depth, dropout=dropout)
        self.high_refine = StripConvStack(feature_dim * 3, depth=max(1, naf_depth // 2), dropout=dropout)
        self.post_refine = NAFStack(feature_dim, depth=max(1, naf_depth - naf_depth // 2), dropout=dropout)
        self.upsampler = ProgressiveUpsampler(feature_dim, stages=upsample_stages, dropout=dropout)
        self.heads = UQHeads(feature_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def _group_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.group_encoders[name](x[:, self.group_indices[name]]) for name in self.group_names]

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input [B, {self.in_channels}, H, W], got {tuple(x.shape)}")
        group_features = self._group_features(x)
        nodes = torch.stack(group_features, dim=1)
        graph_features = self.graph(nodes)
        gated_features = self.gated_fusion(*group_features)
        spectral_features = self.spectral(x)
        # The graph path contains the explicit native-resolution streams. The
        # spectral path carries wavelength/GSD context into the same feature space.
        fused = self.pre_refine(graph_features + gated_features + spectral_features)
        low, high, original_size = self.dwt(fused)
        low = self.spatial(low)
        high = self.high_refine(high)
        fused = self.dwt.inverse(low, high, original_size)
        fused = self.post_refine(fused)
        high_res = self.upsampler(fused)
        mean, logvar = self.heads(high_res)
        skip = F.interpolate(self.skip(x), size=mean.shape[-2:], mode="bilinear", align_corners=False)
        mean = mean + skip
        if return_aux:
            return {
                "mean": mean,
                "logvar": logvar,
                "variance": logvar.exp(),
                "features": fused,
                "high_res": high_res,
                "native_nodes": nodes,
            }
        return mean, logvar
