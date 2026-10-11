import os
import torch
import torch.nn.functional as F
import logging
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def cache_phase1_outputs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Phase 1 Cache Generation on {device}")
    
    # 1. Load the frozen Phase 1 model
    import yaml
    with open('configs/sen2neon_6m.yaml') as f:
        cfg = yaml.safe_load(f)
    model = TriSpecWaveMambaUQ(
        feature_dim=cfg['feature_dim'], spectral_dim=cfg['spectral_dim'],
        naf_depth=cfg['naf_depth'], spatial_depth=cfg['spatial_depth'],
        band_wavelengths_nm=cfg['band_wavelengths_nm'], band_gsd_m=cfg['band_gsd_m'],
        graph_heads=cfg.get('graph_heads', 4)
    ).to(device)
    
    checkpoint_path = "/mnt/pune/checkpoints/sen2neon_6m_phase1/best.pt"
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {}
    for k, v in ckpt['model_state_dict'].items():
        clean_state_dict[k.replace('_orig_mod.', '')] = v
    model.load_state_dict(clean_state_dict)
    model.eval()
    logger.info(f"Loaded Phase 1 model from {checkpoint_path}")
    
    # 2. Setup Directories
    source_dir = "/mnt/pune/sen2neon_patches_16x/train"
    cache_dir = "/mnt/pune/sen2neon_12band_cache/train"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 3. Process Chunks
    files = sorted([f for f in os.listdir(source_dir) if f.endswith('.pt')])
    logger.info(f"Found {len(files)} chunk files to process in {source_dir}")
    
    with torch.no_grad():
        for i, f in enumerate(files):
            source_path = os.path.join(source_dir, f)
            cache_path = os.path.join(cache_dir, f)
            
            if os.path.exists(cache_path):
                logger.info(f"Skipping {f}, already cached.")
                continue
                
            logger.info(f"Processing chunk {i+1}/{len(files)}: {f}")
            data = torch.load(source_path, weights_only=True)
            
            # LR is uint16 in [0, 10000]. Normalize to [0, 1]
            lr_chunk = (data["lr"].float() / 10000.0).clamp(0, 1).to(device)
            
            # We process in mini-batches to avoid OOM
            batch_size = 32
            phase1_out_list = []
            
            for j in range(0, len(lr_chunk), batch_size):
                lr_batch = lr_chunk[j:j+batch_size]
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out, _ = model(lr_batch)
                
                # Move to CPU and cast to float16 to save space on disk
                phase1_out_list.append(out.cpu().half())
            
            phase1_out = torch.cat(phase1_out_list, dim=0)
            
            # Save the cache file
            torch.save({"phase1_out": phase1_out}, cache_path)
            logger.info(f"Saved {cache_path} with shape {phase1_out.shape}")
            
            del data, lr_chunk, phase1_out, phase1_out_list

if __name__ == "__main__":
    cache_phase1_outputs()
