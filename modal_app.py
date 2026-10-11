"""Modal Application for Sentinel-2 L2A Super-Resolution Distillation (Pune AOI).

Distils pretrained OpenSR-SRGAN RGB-NIR teacher into TriSpecWave-Mamba-UQ student.
All heavy workloads run remotely on Modal GPU (NVIDIA A10G) and CPU containers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import modal

# 1. Define Persistent Modal Volume
VOLUME_NAME = "psc26142-pune-distill"
VOLUME_MOUNT = "/mnt/pune"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# 2. Define App
app = modal.App("psc26142-pune-distill")

# 3. Define Container Image
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgdal-dev", "g++", "git")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
        "rasterio>=1.3.0",
        "pystac-client>=0.7.0",
        "planetary-computer>=1.0.0",
        "opensr-srgan",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "tqdm>=4.66.0",
        "tifffile>=2024.2.0",
        "requests>=2.31.0",
        "fastapi>=0.110.0",
        "pydantic>=2.6.0",
        "pytest>=8.0.0",
        "huggingface_hub",
        "datasets",
        "psutil",
        "scikit-image",
        "torchmetrics"
    )
    .add_local_dir(".", remote_path="/root/srm_repo")
)


def _setup_container_paths():
    """Ensure repository root is at the head of sys.path."""
    repo_path = "/root/srm_repo"
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    if os.path.exists(repo_path):
        os.chdir(repo_path)

    # Ensure volume directory structure exists
    for sub in ["raw", "scenes", "patches", "teacher_cache", "checkpoints", "logs", "metrics", "outputs"]:
        os.makedirs(f"{VOLUME_MOUNT}/{sub}", exist_ok=True)


# ==============================================================================
# Remote Function 1: Environment & Pre-flight Verification
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
)
def verify_environment():
    """Check GPU availability, PyTorch CUDA build, packages, and Volume mount."""
    _setup_container_paths()
    import torch
    import rasterio
    from opensr_srgan import load_inference_model
    from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0
    )

    print("=" * 60)
    print("MODAL ENVIRONMENT VERIFICATION (PUNE DISTILLATION)")
    print("=" * 60)
    print(f"Python Version:       {sys.version.split()[0]}")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"CUDA Available:       {torch.cuda.is_available()}")
    print(f"Device Name:          {device_name} ({vram_gb:.1f} GB VRAM)")
    print(f"Rasterio Version:     {rasterio.__version__}")
    print(f"Modal Volume Mount:   {VOLUME_MOUNT}")

    # Test student instantiation
    student = TriSpecWaveMambaUQ(
        in_channels=12,
        out_channels=12,
        feature_dim=32,
        spectral_dim=32,
        upsample_stages=2,
    ).cuda()
    student_params = sum(p.numel() for p in student.parameters())
    print(f"TriSpecWave-Mamba-UQ: Successfully initialized ({student_params:,} parameters)")

    # Test OpenSR-SRGAN teacher instantiation
    print("Loading OpenSR-SRGAN RGB-NIR teacher model...")
    teacher = load_inference_model("RGB-NIR").cuda().eval()
    dummy_in = torch.rand(1, 4, 64, 64, device="cuda")
    with torch.no_grad():
        dummy_out = teacher.predict_step(dummy_in)
    print(f"OpenSR-SRGAN Teacher: Successfully loaded (Output shape: {list(dummy_out.shape)})")

    volume.commit()
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION: ALL CHECKS PASSED!")
    print("=" * 60)
    return {
        "device": device_name,
        "vram_gb": round(vram_gb, 2),
        "student_params": student_params,
        "teacher_output_shape": list(dummy_out.shape),
    }


# ==============================================================================
# Remote Function 2: Download Sentinel-2 L2A Pune Scenes
# ==============================================================================
@app.function(
    image=gpu_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
)
def download_pune_data(radius_km: float = 10.0, max_scenes: int = 5):
    """Search STAC and download 12-band Sentinel-2 L2A GeoTIFFs for Pune AOI."""
    _setup_container_paths()
    from scripts.download_pune_sentinel2 import download_pune_data as _download

    print(f"Initiating remote download of Sentinel-2 L2A scenes around Pune ({radius_km} km radius)...")
    manifest = _download(
        raw_dir=f"{VOLUME_MOUNT}/raw",
        lat=18.4388,
        lon=73.7728,
        radius_km=radius_km,
        max_scenes=max_scenes,
    )
    volume.commit()
    print(f"Successfully downloaded {len(manifest)} scenes to {VOLUME_MOUNT}/raw.")
    return manifest


# ==============================================================================
# Remote Function 3: Prepare Patches
# ==============================================================================
@app.function(
    image=gpu_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=1800,
)
def prepare_pune_patches(max_patches: int = 2000, patch_size: int = 64, stride: int = 32, clean_existing: bool = True):
    """Extract 64x64 patches, filter clouds/shadows via SCL, and write manifest.csv."""
    _setup_container_paths()
    from scripts.prepare_pune_patches import prepare_pune_patches as _prepare

    if clean_existing:
        cache_dir = Path(f"{VOLUME_MOUNT}/teacher_cache")
        if cache_dir.exists():
            print(f"Purging old bloated files from {cache_dir} to reclaim 92GB storage...")
            for f in cache_dir.glob("*.pt"):
                try:
                    f.unlink()
                except Exception:
                    pass

    print(f"Preparing up to {max_patches} Sentinel-2 L2A 64x64 patches (stride={stride})...")
    manifest_path = _prepare(
        raw_dir=f"{VOLUME_MOUNT}/raw",
        patches_dir=f"{VOLUME_MOUNT}/patches",
        patch_size=patch_size,
        stride=stride,
        max_patches=max_patches,
        clean_existing=clean_existing,
    )
    volume.commit()
    print(f"Patches successfully prepared. Manifest committed to volume at {manifest_path}.")
    return manifest_path


# ==============================================================================
# Remote Function 4: Cache OpenSR-SRGAN Teacher Targets
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="A10G",
    cpu=8,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
)
def cache_teacher_targets(max_patches: int = 2000, batch_size: int = 32):
    """Run frozen OpenSR-SRGAN RGB-NIR teacher and cache targets to /mnt/pune/teacher_cache/."""
    _setup_container_paths()
    from scripts.cache_srgan_targets import cache_teacher_targets as _cache

    print(f"Running OpenSR-SRGAN teacher caching on NVIDIA A10G with 8 vCPUs (batch_size={batch_size}, max_patches={max_patches})...")
    total_cached = _cache(
        patches_dir=f"{VOLUME_MOUNT}/patches",
        cache_dir=f"{VOLUME_MOUNT}/teacher_cache",
        max_patches=max_patches,
        batch_size=batch_size,
    )
    volume.commit()
    print(f"Teacher target caching complete. Total cached targets on volume: {total_cached}.")
    return total_cached


# ==============================================================================
# Remote Function 5: Student Distillation Training
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="A10G",
    cpu=8,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600 * 6,
)
def distill_student(max_steps: int = 8000, num_patches: int | None = None, resume: bool = False, warm_start: bool = True):
    """Execute TriSpecWave-Mamba-UQ student distillation training loop on NVIDIA A10G."""
    _setup_container_paths()
    from scripts.distill_srgan import distill_student as _distill

    print(f"Starting remote student distillation training on NVIDIA A10G for {max_steps} steps (warm_start={warm_start}, resume={resume})...")
    result = _distill(
        config_path="configs/distill_pune.yaml",
        max_steps=max_steps,
        num_patches=num_patches,
        resume=resume,
        warm_start=warm_start,
    )
    volume.commit()
    print(f"Distillation training finished! Best checkpoint committed to volume: {result['best_checkpoint']}")
    return result


# ==============================================================================
# Remote Function: 1-Patch Diagnostic Test
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="T4",
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
)
def run_one_patch_test(max_steps: int = 1500):
    """Run a single-patch diagnostic test on a T4 GPU to verify edge-learning capacity."""
    _setup_container_paths()
    from scripts.train_one_patch import run_one_patch_test as _run_test

    print(f"Starting 1-patch diagnostic test on T4 for {max_steps} steps...")
    result = _run_test(
        config_path="configs/distill_pune.yaml",
        max_steps=max_steps,
    )
    print(f"1-Patch test completed! Final PSNR: {result['final_psnr']:.2f} dB")
    return result


# ==============================================================================
# Remote Function 6: Evaluation & Reporting
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=1800,
)
def evaluate_distillation(samples: int = 50, checkpoint_path: str | None = None, split: str = "test"):
    """Evaluate student against Bicubic baseline and SRGAN teacher on specified split (default 'test')."""
    _setup_container_paths()
    from scripts.evaluate_distillation import evaluate_model as _eval

    print(f"Executing remote evaluation on {samples} '{split}' patches (checkpoint={checkpoint_path})...")
    report = _eval(
        config_path="configs/distill_pune.yaml",
        checkpoint_path=checkpoint_path,
        num_eval_samples=samples,
        split=split,
    )
    volume.commit()
    print("Remote evaluation completed and visual plots committed to volume.")
    return report


# ==============================================================================
# Remote Function: Stage 5 Detail Head Training
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
)
def train_stage5_detail(steps: int = 2000, lr: float = 5e-4):
    """Train the NAF Detail Head on cached features from frozen backbone."""
    _setup_container_paths()
    from scripts.train_detail_head import train_stage5_detail as _train_stage5

    print(f"Starting Stage 5 Detail Head training on A10G (steps={steps}, lr={lr})...")
    _train_stage5(steps=steps, lr=lr)
    volume.commit()
    print("Stage 5 Detail Head training completed.")
    return True

# ==============================================================================
# Remote Function: Stage 5 Evaluation
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=1800,
)
def evaluate_stage5(samples: int = 50):
    """Evaluate the Stage 5 (Bicubic + Detail) model."""
    _setup_container_paths()
    from scripts.evaluate_stage5 import evaluate_stage5 as _eval_stage5

    print(f"Starting Stage 5 evaluation on A10G (samples={samples})...")
    _eval_stage5(samples=samples)
    volume.commit()
    print("Stage 5 evaluation completed.")
    return True

# ==============================================================================
# Remote Function: Download SEN2NEON Dataset
# ==============================================================================
@app.function(
    image=gpu_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
)
def download_sen2neon_data():
    """Download the SEN2NEON 2.5m dataset directly to the Modal volume."""
    _setup_container_paths()
    from scripts.download_sen2neon import download_sen2neon as _download

    print("Starting download of SEN2NEON dataset from HuggingFace to Modal volume...")
    _download(target_dir=f"{VOLUME_MOUNT}/sen2neon")
    volume.commit()
    print("Download complete and committed to volume.")
    return True


# ==============================================================================
# Remote Function: Stage 6 SEN2NEON Training
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=14400,
)
def train_stage6_sen2neon(steps: int = 10000):
    """Train the base model end-to-end on real SEN2NEON ground truth."""
    _setup_container_paths()
    from scripts.train_sen2neon import train_stage6_sen2neon as _train

    print(f"Starting Stage 6 SEN2NEON Training on A10G (steps={steps})...")
    _train(steps=steps)
    volume.commit()
    print("Stage 6 training completed.")
    return True

# ==============================================================================
# Remote Function: Stage 6 SEN2NEON Evaluation
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=1800,
)
def evaluate_stage6_sen2neon(samples: int = 50):
    """Evaluate the 12-band model on the SEN2NEON validation set."""
    _setup_container_paths()
    from scripts.evaluate_sen2neon import evaluate_sen2neon as _eval

    print(f"Starting Stage 6 SEN2NEON Evaluation on A10G (samples={samples})...")
    _eval(samples=samples)
    volume.commit()
    print("Stage 6 evaluation completed.")
    return True

# ==============================================================================
# Remote Function 7: Run Shape & Unit Tests Remotely
# ==============================================================================
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
)
def run_remote_tests():
    """Run pytest suite remotely on Modal."""
    _setup_container_paths()
    import pytest

    print("Running automated shape, band order, and architecture test suite remotely...")
    retcode = pytest.main(["-v", "tests/test_distillation_shapes.py"])
    if retcode != 0:
        raise RuntimeError(f"Pytest failed with exit code {retcode}")
    print("All remote unit tests passed successfully!")
    return "All tests passed"


# ==============================================================================
# Remote Function 8: List Volume Contents
# ==============================================================================
@app.function(
    image=gpu_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=300,
)
def list_volume_outputs():
    """List persistent files and directory hierarchy on Modal volume."""
    _setup_container_paths()
    volume_root = Path(VOLUME_MOUNT)
    summary = {}
    for sub in ["raw", "scenes", "patches", "teacher_cache", "checkpoints", "logs", "metrics", "outputs"]:
        sub_path = volume_root / sub
        if sub_path.exists():
            files = list(sub_path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_size_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
            summary[sub] = {"files": file_count, "size_mb": round(total_size_mb, 2)}
        else:
            summary[sub] = {"files": 0, "size_mb": 0.0}

    print("=" * 50)
    print("MODAL VOLUME STORAGE SUMMARY (/mnt/pune)")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:15s}: {v['files']:5d} files ({v['size_mb']:8.2f} MB)")
    print("=" * 50)
    return summary

@app.function(
    image=gpu_image,
    gpu='H100',
    cpu=8,
    memory=65536,
    volumes={VOLUME_MOUNT: volume},
    timeout=28800,
)
def train_stage7_sen2neon_6m(updates: int = 5000, profile_only: bool = False):
    '''Train 6.17M param model from scratch on SEN2NEON ground truth.'''
    _setup_container_paths()
    from scripts.train_sen2neon_6m import train_sen2neon_6m as _train
    
    import yaml
    with open('configs/sen2neon_6m.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['max_updates'] = updates
    with open('configs/sen2neon_6m.yaml', 'w') as f:
        yaml.safe_dump(cfg, f)
        
    print(f'Starting Stage 7 SEN2NEON 6.17M Training on H100 (updates={updates})...')
    _train(profile_only=profile_only)
    volume.commit()
    print('Stage 7 training complete.')


@app.function(
    image=gpu_image,
    cpu=8,
    memory=65536,
    volumes={VOLUME_MOUNT: volume},
    timeout=7200,
)
def extract_sen2neon_patches():
    '''CPU job to pre-extract 64x64 patches into .pt chunks.'''
    _setup_container_paths()
    from scripts.extract_sen2neon_patches import extract_patches
    extract_patches()
    volume.commit()
    print('Extraction complete.')


@app.function(
    image=gpu_image,
    gpu='H100',
    cpu=8,
    memory=65536,
    volumes={VOLUME_MOUNT: volume},
    timeout=14400,
)
def train_stage7_patches():
    '''Ultra-fast H100 training on pre-extracted patches.'''
    _setup_container_paths()
    from scripts.train_sen2neon_6m_patches import train_patches
    train_patches()
    volume.commit()
    print('Stage 7 patch training complete.')


@app.function(
    image=gpu_image,
    gpu='H100',
    cpu=8,
    memory=262144,
    volumes={VOLUME_MOUNT: volume},
    timeout=14400,
)
def train_stage7_ram():
    '''Maximum throughput H100 training loading 120GB into RAM.'''
    _setup_container_paths()
    from scripts.train_sen2neon_6m_ram import train_ram
    train_ram()
    volume.commit()
    print('Stage 7 RAM training complete.')



@app.function(
    image=gpu_image,
    gpu='L4',
    cpu=4,
    memory=65536,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
)
def evaluate_stage7_ram():
    '''Runs SEN2NEON benchmark metrics (PSNR, SSIM, SAM) on the validation set using an L4 GPU.'''
    _setup_container_paths()
    from scripts.evaluate_sen2neon_benchmark import run_benchmark
    run_benchmark()
    volume.commit()
    print('Benchmark evaluation complete.')

@app.function(
    image=gpu_image,
    gpu='T4',
    cpu=4,
    memory=16384,
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
)
def generate_good_visual_endpoint():
    '''Generates a visual comparison grid bypassing blank patches.'''
    _setup_container_paths()
    from scripts.generate_good_visual import generate_good_visual
    generate_good_visual()
    volume.commit()
    print('Visual generated.')

@app.function(
    image=gpu_image,
    gpu='T4',
    cpu=4,
    memory=16384,
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
)
def evaluate_pune_generalization():
    '''Generates a visual comparison grid on Pune data.'''
    _setup_container_paths()
    from scripts.evaluate_6m_on_pune import generate_pune_comparison
    generate_pune_comparison()
    volume.commit()
    print('Pune visual generated.')

@app.function(
    image=gpu_image,
    gpu='L4',
    cpu=4,
    memory=16384,
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
)
def generate_6_patch_comparison_endpoint():
    '''Generates a 6-row by 4-column visual test comparison of LR vs Bicubic vs 6M vs HR on validation patches.'''
    _setup_container_paths()
    from scripts.generate_6_patch_comparison import generate_six_comparisons
    generate_six_comparisons()
    volume.commit()
    print('6-Patch visual comparison generated.')

@app.function(
    image=gpu_image,
    gpu='H100',
    cpu=8,
    memory=262144,
    volumes={VOLUME_MOUNT: volume},
    timeout=14400,
)
def train_ram_phase1():
    '''Maximum throughput H100 training with Phase 1 Fix (Shifted L1 + Nodata Mask).'''
    _setup_container_paths()
    from scripts.train_sen2neon_6m_phase1 import train_phase1
    train_phase1()
    volume.commit()
    print('Phase 1 training complete.')

@app.function(
    image=gpu_image,
    gpu='L4',
    volumes={VOLUME_MOUNT: volume},
)
def evaluate_bicubic_baseline():
    '''Calculates the pure bicubic PSNR on the dataset chunk.'''
    _setup_container_paths()
    import torch
    import torch.nn.functional as F
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    
    print("Loading validation chunk...")
    val_data = torch.load("/mnt/pune/sen2neon/ram_cache/val_chunk_0000.pt")
    lr, hr, mask = val_data
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    bicubic = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)
    bicubic = bicubic.clamp(0, 1)
    hr = hr.clamp(0, 1)
    
    psnr = psnr_metric(bicubic, hr)
    ssim = ssim_metric(bicubic, hr)
    
    print(f"BICUBIC PSNR: {psnr.item():.4f}")
    print(f"BICUBIC SSIM: {ssim.item():.4f}")

@app.function(image=gpu_image, volumes={VOLUME_MOUNT: volume})
def get_target_image_endpoint():
    _setup_container_paths()
    import scripts.get_target_image as get_target_image
    get_target_image.get_target()
    volume.commit()
    print('Target image cropped.')


@app.function(image=gpu_image, gpu='L4', volumes={VOLUME_MOUNT: volume})
def infer_target_pune_endpoint():
    _setup_container_paths()
    import scripts.infer_target_pune as infer_target_pune
    infer_target_pune.infer_target()
    volume.commit()
    print('Target inference completed.')


@app.function(image=gpu_image, gpu='L4', volumes={VOLUME_MOUNT: volume})
def evaluate_official_benchmark_endpoint():
    _setup_container_paths()
    import scripts.evaluate_official_benchmark as eval_bench
    eval_bench.run_evaluation()





@app.function(
    image=gpu_image,
    gpu='A10G', 
    timeout=3600,
    volumes={VOLUME_MOUNT: volume}
)
def generate_phase1_cache_endpoint():
    _setup_container_paths()
    from scripts.cache_phase1_outputs import cache_phase1_outputs
    cache_phase1_outputs()

@app.function(
    image=gpu_image,
    gpu='L4', 
    timeout=3600,
    volumes={VOLUME_MOUNT: volume}
)
def train_unet_adapter_endpoint():
    _setup_container_paths()
    from scripts.train_unet_adapter import train_adapter
    train_adapter()
