import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import Dataset, DataLoader
import time
import yaml

from models.rgbn_unet_adapter import RGBNUNetAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Losses ---
def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))

def sobel_gradients(img):
    kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img.device).view(1, 1, 3, 3).expand(img.size(1), 1, 3, 3)
    kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=img.device).view(1, 1, 3, 3).expand(img.size(1), 1, 3, 3)
    grad_x = F.conv2d(img, kernel_x, padding=1, groups=img.size(1))
    grad_y = F.conv2d(img, kernel_y, padding=1, groups=img.size(1))
    return grad_x, grad_y

def gradient_loss(pred, target):
    pred_x, pred_y = sobel_gradients(pred)
    target_x, target_y = sobel_gradients(target)
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)

def sam_loss(pred, target, eps=1e-8):
    dot_product = torch.sum(pred * target, dim=1)
    norm_pred = torch.norm(pred, dim=1)
    norm_target = torch.norm(target, dim=1)
    cos_theta = (dot_product + eps) / (norm_pred * norm_target + eps)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.mean(torch.acos(cos_theta))

def composite_loss(pred, target, mask=None):
    if mask is not None:
        pred = pred * mask
        target = target * mask
        
    c_loss = charbonnier_loss(pred, target)
    g_loss = gradient_loss(pred, target)
    s_loss = sam_loss(pred, target)
    
    # Weights based on critique agent's recommendation
    # (Removed Laplacian to prevent clashes, kept SAM light)
    total = c_loss + (0.15 * g_loss) + (0.05 * s_loss)
    return total, c_loss.item(), g_loss.item(), s_loss.item()

# --- Dataset ---
class CachedPhase1Dataset(Dataset):
    def __init__(self, source_dir, cache_dir):
        self.source_dir = source_dir
        self.cache_dir = cache_dir
        
        self.files = sorted([f for f in os.listdir(source_dir) if f.endswith('.pt')])
        
        # Build index mapping global index to (file_idx, patch_idx)
        self.index_map = []
        logger.info("Building dataset index...")
        for f_idx, f in enumerate(self.files):
            num_patches = 672 if f == "chunk_0032.pt" else 1000
            for p_idx in range(num_patches):
                self.index_map.append((f_idx, p_idx))
                
        logger.info(f"Dataset indexed. Total patches: {len(self.index_map)}")
        
        # We cache the CURRENT open file to avoid reloading it constantly
        self.current_file_idx = -1
        self.current_source_data = None
        self.current_cache_data = None
        
    def __len__(self):
        return len(self.index_map)
        
    def __getitem__(self, idx):
        f_idx, p_idx = self.index_map[idx]
        
        if self.current_file_idx != f_idx:
            filename = self.files[f_idx]
            self.current_source_data = torch.load(os.path.join(self.source_dir, filename), weights_only=True)
            self.current_cache_data = torch.load(os.path.join(self.cache_dir, filename), weights_only=True)
            self.current_file_idx = f_idx
            
        lr_orig = (self.current_source_data["lr"][p_idx].float() / 10000.0).clamp(0, 1)
        hr_orig = (self.current_source_data["hr"][p_idx].float() / 10000.0).clamp(0, 1)
        phase1_out = self.current_cache_data["phase1_out"][p_idx].float()
        
        # We need bicubic of LR
        bicubic_in = F.interpolate(lr_orig.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)
        
        # Extract RGBN for HR
        rgbn_idx = [1, 2, 3, 7]
        hr_rgbn = hr_orig[rgbn_idx, :, :]
        
        # Calculate validity mask (where target > 0)
        mask = (hr_orig.sum(dim=0, keepdim=True) > 0).float()
        
        return phase1_out, bicubic_in, hr_rgbn, mask

# --- Training Loop ---
def train_adapter():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting UNet Adapter Training on {device}")
    
    source_dir = "/mnt/pune/sen2neon_patches_16x/train"
    cache_dir = "/mnt/pune/sen2neon_12band_cache/train"
    
    if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
        logger.error("Phase 1 Cache not found! Run cache_phase1_outputs.py first.")
        return
        
    dataset = CachedPhase1Dataset(source_dir, cache_dir)
    # Using small batch size to fit on any GPU (even T4)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    
    model = RGBNUNetAdapter(in_channels=24, out_channels=4).to(device)
    
    # Check if we should resume from epoch 1
    checkpoint_path = "/mnt/pune/checkpoints/sen2neon_unet_adapter/epoch_1.pt"
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        start_epoch = 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    EPOCHS = 3
    logger.info("Starting Training Loop...")
    
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        
        for step, (phase1_out, bicubic_in, hr_rgbn, mask) in enumerate(loader):
            phase1_out = phase1_out.to(device, non_blocking=True)
            bicubic_in = bicubic_in.to(device, non_blocking=True)
            hr_rgbn = hr_rgbn.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_rgbn = model(phase1_out, bicubic_in)
                loss, c_loss, g_loss, s_loss = composite_loss(pred_rgbn, hr_rgbn, mask)
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % 50 == 0:
                logger.info(f"Ep {epoch} | Step {step}/{len(loader)} | Loss: {loss.item():.4f} (C:{c_loss:.4f} G:{g_loss:.4f} S:{s_loss:.4f})")
                
        # Save checkpoint
        os.makedirs("/mnt/pune/checkpoints/sen2neon_unet_adapter", exist_ok=True)
        torch.save(model.state_dict(), f"/mnt/pune/checkpoints/sen2neon_unet_adapter/epoch_{epoch}.pt")
        logger.info(f"Epoch {epoch} finished. Average Loss: {epoch_loss / len(loader):.4f}")
        
if __name__ == "__main__":
    train_adapter()
