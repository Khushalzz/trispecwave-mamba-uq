import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBNUNetAdapter(nn.Module):
    def __init__(self, in_channels=24, out_channels=4):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Output layer
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Zero-initialize the final layer
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, phase1_out, bicubic_in):
        # Concatenate phase1 full output and bicubic full input
        # Expecting phase1_out: [B, 12, H, W]
        # Expecting bicubic_in: [B, 12, H, W]
        x = torch.cat([phase1_out, bicubic_in], dim=1) # [B, 24, H, W]
        
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool1(e1)))
        e3 = F.relu(self.enc3(self.pool2(e2)))
        
        d1 = self.up1(e3)
        # Skip connection 1
        if d1.shape != e2.shape:
            d1 = F.interpolate(d1, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d1 = F.relu(self.dec1(torch.cat([d1, e2], dim=1)))
        
        d2 = self.up2(d1)
        # Skip connection 2
        if d2.shape != e1.shape:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.relu(self.dec2(torch.cat([d2, e1], dim=1)))
        
        residual = self.out(d2)
        
        # Extract RGBN bands from phase1_out (indices 1, 2, 3, 7)
        rgbn_idx = [1, 2, 3, 7]
        phase1_rgbn = phase1_out[:, rgbn_idx, :, :]
        
        # Add residual and clamp to physical reflectance bounds [0, 1]
        final_rgbn = torch.clamp(phase1_rgbn + residual, 0.0, 1.0)
        
        return final_rgbn
