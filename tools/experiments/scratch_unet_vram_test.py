import torch
import torch.nn as nn

class UNetAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(4, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Conv2d(64, 128, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(self.pool1(e1)))
        e3 = torch.relu(self.enc3(self.pool2(e2)))
        d1 = torch.relu(self.dec1(torch.cat([self.up1(e3), e2], dim=1)))
        d2 = torch.relu(self.dec2(torch.cat([self.up2(d1), e1], dim=1)))
        return self.out(d2)

if torch.cuda.is_available():
    model = UNetAdapter().cuda()
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params / 1e6:.2f} M")

    # Simulate a training batch (Batch Size 16, 4 Channels, 256x256 HR patches)
    x = torch.randn(16, 4, 256, 256, device='cuda', requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    out = model(x)
    loss = out.sum()
    loss.backward()
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak VRAM for BS=16, 256x256 patches: {peak_vram:.2f} MB")
else:
    print("No CUDA device found.")
