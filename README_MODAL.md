# OpenSR-SRGAN to TriSpecWave-Mamba-UQ Distillation on Modal

End-to-end framework to distil the pretrained **ESA OpenSR-SRGAN RGB-NIR** teacher model into the 12-band **TriSpecWave-Mamba-UQ** student network on Sentinel-2 Level-2A surface reflectance data over **Pune, Maharashtra, India**.

All heavy compute (STAC downloading, patch extraction, teacher caching, student training, and evaluation) runs **remotely on Modal GPU/CPU containers** backed by a persistent Modal Volume. The local Windows machine only drives the Modal CLI.

---

## 1. Prerequisites & Modal Setup

### 1.1 Install Modal CLI
```powershell
python -m pip install modal
```

### 1.2 Authenticate Modal
```powershell
modal setup
```

### 1.3 Create Persistent Modal Volume
```powershell
modal volume create psc26142-pune-distill --version=2
```
Mounted inside remote Modal containers at: `/mnt/pune`

---

## 2. Windows Launcher (`run.bat`)

All operations can be driven using the Windows launcher:

| Command | Description |
| :--- | :--- |
| `run.bat setup` | Verify CLI auth, create volume `psc26142-pune-distill`, verify remote GPU environment |
| `run.bat verify` | Run remote GPU environment check and automated PyTorch shape tests |
| `run.bat download` | Remotely query STAC and download 12-band Pune Sentinel-2 L2A scenes |
| `run.bat prepare [N]` | Remotely crop 64x64 patches, apply SCL cloud filter, generate `manifest.csv` (default: 500) |
| `run.bat cache [N] [--detach]` | Remotely run frozen OpenSR-SRGAN teacher and cache targets on NVIDIA A10G |
| `run.bat distill [STEPS] [--detach]` | Remotely train TriSpecWave-Mamba-UQ student on NVIDIA A10G (default: 3000 steps) |
| `run.bat evaluate` | Benchmark against bicubic baseline and teacher; generate metrics & plots |
| `run.bat all` | Two-phase pipeline: 50-patch sanity test, then full 500-patch distillation |
| `run.bat pull` | Download `best.pt`, `metrics.json`, and comparison plots to `./outputs/` |
| `run.bat volume` | View persistent storage summary across `/mnt/pune` directories |
| `run.bat logs` | List active Modal apps and tail logs |

---

## 3. Dataset & Pune AOI

- **Location**: Pune, Maharashtra, India
- **Coordinates**: Latitude `18.4388° N`, Longitude `73.7728° E`
- **Area of Interest**: ~10 km radius bounding box: `[73.68° E, 18.35° N, 73.87° E, 18.53° N]`
- **Catalogue**: Microsoft Planetary Computer STAC (`collection: sentinel-2-l2a`) with fallback to Copernicus Data Space STAC.
- **Acquisitions**: 5 distinct months (Jan, Feb, Mar, Apr, May 2024), all with `< 0.05%` cloud cover (Tile `T43QCA`).
- **Data Splitting**: Partitioned strictly by acquisition date:
  - 3 scenes for `train`
  - 1 scene for `val`
  - 1 scene for `test`

---

## 4. Band Ordering & Spatial Layout

**12 Bands (excluding B10)**:
```
Index:   0    1    2    3    4    5    6    7    8    9    10   11
Band:   B1   B2   B3   B4   B5   B6   B7   B8   B8A  B9   B11  B12
GSD:    60m  10m  10m  10m  20m  20m  20m  10m  20m  60m  20m  20m
Wl(nm): 443  490  560  665  705  740  783  842  865  945  1610 2190
```

### Native Groups
- **10 m Native**: `[1, 2, 3, 7]` \(\rightarrow\) `[B2, B3, B4, B8]` (Supervised by OpenSR-SRGAN RGB-NIR teacher)
- **20 m Native**: `[4, 5, 6, 8, 10, 11]` \(\rightarrow\) `[B5, B6, B7, B8A, B11, B12]`
- **60 m Native**: `[0, 9]` \(\rightarrow\) `[B1, B9]`

---

## 5. Tensor Shapes & Distillation Architecture

```
Student Input x:                [B=1, 12, 64, 64]   (float32 reflectance [0, 1])
Teacher Input (bands 1,2,3,7):  [B=1, 4, 64, 64]    (B2, B3, B4, B8)
Teacher predict_step Output:    [B=1, 4, 256, 256]  (2.5m pseudo-target)
Student Forward (hr, logvar):   [B=1, 12, 256, 256] (2.5m super-resolved)
Student Supervised RGBN:        [B=1, 4, 256, 256]  (student_hr[:, [1, 2, 3, 7]])
Degraded Student (MTF/PSF):     [B=1, 12, 64, 64]   (sensor consistency check)
```

### Loss Function
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{teacher}} + 0.25 \cdot \mathcal{L}_{\text{sensor}} + 0.05 \cdot \mathcal{L}_{\text{sam}}$$

- $\mathcal{L}_{\text{teacher}} = \text{L1}(\text{student\_rgbn}, \text{teacher\_hr})$
- $\mathcal{L}_{\text{sensor}} = \text{L1}(\text{degraded\_student}_{12\text{-band}}, x_{12\text{-band}})$ (MTF blur + area downsampling)
- $\mathcal{L}_{\text{sam}} = \text{SAM}(\text{student\_rgbn}, \text{teacher\_hr})$

---

## 6. Modal Volume Directory Structure (`/mnt/pune`)

```
/mnt/pune/
├── raw/                # Downloaded Sentinel-2 L2A GeoTIFFs (B1-B12, SCL)
├── scenes/             # Scene level indices
├── patches/            # 64x64 float32 patches and manifest.csv
├── teacher_cache/      # Cached 4x teacher pseudo-targets (.pt files)
├── checkpoints/        # best.pt and last.pt student weights
├── logs/               # distill.log training logs
├── metrics/            # metrics.json benchmark evaluation report
└── outputs/            # pune_comparison.png, pune_uncertainty.png, pune_student_rgbn.png
```

---

## 7. How to Resume After Interruption

The entire pipeline is built with automatic idempotency:
- `download_pune_data`: Checks `scenes_manifest.json`; skips download if scenes exist.
- `prepare_pune_patches`: Checks `manifest.csv`; skips if patches are already prepared.
- `cache_teacher_targets`: Inspects `/mnt/pune/teacher_cache/{patch_id}.pt` and skips already cached targets.
- `distill_student`: Automatically detects `/mnt/pune/checkpoints/last.pt` and resumes from the exact training step.
- Every function commits the volume before returning via `volume.commit()`.

---

## 8. Downloading Artifacts

To pull trained model weights, evaluation metrics, and comparison images to the local `./outputs/` folder:
```powershell
run.bat pull
```
Or via raw Modal CLI:
```powershell
modal volume get psc26142-pune-distill checkpoints/best.pt ./outputs/best.pt
modal volume get psc26142-pune-distill metrics/metrics.json ./outputs/metrics.json
modal volume get psc26142-pune-distill outputs ./outputs/
```
