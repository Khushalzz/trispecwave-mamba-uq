import logging
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.detail_head import NAFDetailHead
from scripts.train_one_patch import gradient_loss

import csv

class TeacherCacheDataset(Dataset):
    """Loads pre-cached (input_patch, teacher_output) pairs into RAM for zero-latency training."""

    def __init__(self, cache_dir, split: str = "train", max_samples=None) -> None:
        self.cache_dir = Path(cache_dir)
        
        # 1. Gather paths
        sample_paths = []
        manifest_csv = self.cache_dir.parent / "patches" / "manifest.csv"

        if manifest_csv.exists():
            with open(manifest_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("split") == split:
                        sample_paths.append(self.cache_dir / f"{r['patch_id']}.pt")
        else:
            sample_paths = sorted(list(self.cache_dir.glob("*.pt")))

        if max_samples and max_samples < len(sample_paths):
            sample_paths = sample_paths[:max_samples]

        logger.info(f"Dataset split='{split}': loading {len(sample_paths)} patches into RAM from {cache_dir}...")

        # 2. Preload into memory
        self.data = []
        for f in sample_paths:
            pt = torch.load(f, map_location="cpu", weights_only=False)
            x = pt["input_patch"].float().clone()
            y = pt["teacher_output"].float().clone()
            self.data.append((x, y))
            
        logger.info(f"Dataset split='{split}': finished loading {len(self.data)} patches into RAM.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_detail")

TEACHER_BANDS = [1, 2, 3, 7]

class CachedFeatureDataset(Dataset):
    def __init__(self, features, targets, bicubic_bases):
        self.features = features
        self.targets = targets
        self.bicubic_bases = bicubic_bases
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            "high_res": self.features[idx],
            "target": self.targets[idx],
            "bicubic": self.bicubic_bases[idx]
        }

def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))

def train_stage5_detail(config_path="configs/distill_pune.yaml", steps=2000, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Stage 5 Detail Head training on {device} (On-the-fly)...")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 1. Load backbone and freeze
    model = TriSpecWaveMambaUQ(
        in_channels=cfg.get("in_channels", 12),
        out_channels=cfg.get("out_channels", 12),
        feature_dim=cfg.get("feature_dim", 64),
        spectral_dim=cfg.get("spectral_dim", 64),
        naf_depth=cfg.get("naf_depth", 4),
        spatial_depth=cfg.get("spatial_depth", 2),
        graph_heads=cfg.get("graph_heads", 4),
        band_wavelengths_nm=cfg.get("band_wavelengths_nm"),
        band_gsd_m=cfg.get("band_gsd_m"),
    ).to(device)
    
    ckpt_path = "/mnt/pune/checkpoints/edge_finetune/best.pt"
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Missing backbone checkpoint at {ckpt_path}")
        
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # 2. Build Datasets
    cache_dir = cfg.get("paths", {}).get("teacher_cache_dir", "/mnt/pune/teacher_cache")
    train_ds = TeacherCacheDataset(cache_dir, split="train")
    val_ds = TeacherCacheDataset(cache_dir, split="val")
    
    # Use num_workers=0 since dataset is in RAM
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True, num_workers=0)
    train_iter = iter(train_loader)
    
    # 3. Initialize Detail Head
    detail_head = NAFDetailHead(in_channels=64, out_channels=4, hidden_dim=64).to(device)
    detail_head.train()
    
    optimizer = torch.optim.AdamW(detail_head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    logger.info(f"Training Detail Head for {steps} steps...")
    start_time = time.time()
    best_val_loss = float("inf")
    
    for step in range(1, steps + 1):
        try:
            x, teacher_hr = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, teacher_hr = next(train_iter)
            
        x = x.to(device).float()
        teacher_hr = teacher_hr.to(device)
        
        with torch.no_grad():
            aux = model(x, return_aux=True)
            high_res = aux["high_res"]
            bicubic = F.interpolate(x[:, TEACHER_BANDS], size=(256, 256), mode="bicubic", align_corners=False).clamp(0, 1)
        
        optimizer.zero_grad(set_to_none=True)
        
        pred_detail = detail_head(high_res)
        detail_target = teacher_hr - bicubic
        
        l_charb = charbonnier_loss(pred_detail, detail_target)
        
        reconstructed = (bicubic + pred_detail).clamp(0, 1)
        l_grad = gradient_loss(reconstructed, teacher_hr)
        
        loss = l_charb + 0.20 * l_grad
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0 or step == steps:
            logger.info(f"Step [{step:4d}/{steps}] Loss: {loss.item():.4f} (Charb: {l_charb.item():.4f}, Grad: {l_grad.item():.4f})")
            
        if step % 500 == 0 or step == steps:
            # Quick val
            detail_head.eval()
            val_loss = 0.0
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
            with torch.no_grad():
                for v_x, v_teacher in val_loader:
                    v_x = v_x.to(device).float()
                    v_teacher = v_teacher.to(device)
                    
                    v_aux = model(v_x, return_aux=True)
                    v_high = v_aux["high_res"]
                    v_bicubic = F.interpolate(v_x[:, TEACHER_BANDS], size=(256, 256), mode="bicubic", align_corners=False).clamp(0, 1)
                    
                    v_pred = detail_head(v_high)
                    v_detail_target = v_teacher - v_bicubic
                    
                    v_l = charbonnier_loss(v_pred, v_detail_target) + 0.20 * gradient_loss((v_bicubic + v_pred).clamp(0,1), v_teacher)
                    val_loss += v_l.item()
                    
            val_loss /= len(val_loader)
            logger.info(f"--> Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = "/mnt/pune/checkpoints/detail_head_best.pt"
                torch.save(detail_head.state_dict(), save_path)
                logger.info(f"--> Saved best detail head to {save_path}")
            
            detail_head.train()
            
    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.1f}s.")
    
if __name__ == "__main__":
    train_stage5_detail()
