import torch
from torch import nn
from .naf_blocks import NAFBlock

class NAFDetailHead(nn.Module):
    """Predicts high-frequency residual detail [B, 4, 256, 256] from high_res features."""
    def __init__(self, in_channels: int = 64, out_channels: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.blocks = nn.Sequential(
            NAFBlock(in_channels),
            NAFBlock(in_channels),
        )
        self.conv_out = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # Initialize final conv with small weights so initial detail is near zero
        nn.init.zeros_(self.conv_out.bias)
        nn.init.normal_(self.conv_out.weight, std=1e-3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.blocks(features))
