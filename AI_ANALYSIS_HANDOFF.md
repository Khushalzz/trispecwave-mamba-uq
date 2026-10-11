# 🛰️ TriSpecWave-Mamba-UQ: Complete Technical Dossier & Project State

> **Notice for Reviewing AI:**  
> This dossier details exactly **what was implemented, tested, and achieved**, followed by **the exact state of the model, metrics, and codebase right now**.  
> All relevant code, trained weights (`best.pt`), evaluation logs (`metrics.json`), and high-resolution visual plots are included in this package.

---

## Part 1: Project Objective & Problem Scope

- **Problem Statement:** Smart India Hackathon (SIH) Problem Statement 2 (PS-2) — Multi-spectral Super-Resolution of **Sentinel-2 L2A** satellite imagery from **10m to 2.5m Ground Sampling Distance (4× spatial upscaling)**.
- **Spectral Scope:** All **12 native spectral bands** (excluding B10 atmosphere):
  - **10m bands:** B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
  - **20m bands:** B5 (RedEdge-1), B6 (RedEdge-2), B7 (RedEdge-3), B8A (Narrow NIR), B11 (SWIR-1), B12 (SWIR-2)
  - **60m bands:** B1 (Coastal Aerosol), B9 (Water Vapor)
- **Core Requirements:**
  1. Full 12-band simultaneous super-resolution (not just RGB-NIR).
  2. Physical sensor consistency: downscaling the 2.5m output through Sentinel-2's Modulation Transfer Function / Point Spread Function (MTF/PSF) must reconstruct the original 10m observation.
  3. Spectral fidelity: no color hallucination (measured via Spectral Angle Mapper, SAM).
  4. Uncertainty Quantification (UQ): pixel-level aleatoric uncertainty ($\sigma$) via heteroscedastic Gaussian Negative Log-Likelihood (NLL).
  5. Real-time inference capability on commodity GPUs ($\le 50$ ms per $256 \times 256$ patch).

---

## Part 2: What We Did (Chronological Execution Log)

### 1. Data Pipeline & Zero-Latency RAM Caching
- **Source:** Acquired Sentinel-2 L2A tiles over Pune, Maharashtra (10 km radius AOI).
- **Splits:** Partitioned strictly by acquisition date into Train (1,400 patches), Validation (100 patches), and held-out Test (50 patches).
- **SCL Cloud Masking:** Filtered invalid pixels, clouds, and cloud shadows using Scene Classification Layer (SCL) masks (max cloud threshold $\le 2\%$).
- **RAM Caching:** Modal network volume FUSE latency previously bottlenecked training to 0.3 steps/s. We overhauled `TeacherCacheDataset` to load all 1,400 training and 100 validation patches into host RAM as pre-allocated torch tensors. Training throughput immediately jumped from **0.3 steps/s to 3.2 steps/s**.

### 2. Architecture Design (`TriSpecWave-Mamba-UQ`, 1.117M Parameters)
Located in `models/trispecwave_mamba_uq.py`:
- **Group Encoders:** Separate sub-networks processing native 10m, 20m, and 60m band groups, fused via `CrossResolutionGatedFusion` and `SpectralGraphMixer`.
- **SpectralMamba:** 1D selective state space model across the 12 spectral channels, injected with explicit physical wavelength (nm) and native resolution (GSD in meters) positional embeddings.
- **Haar Wavelet Decomposition (`HaarDWT2D`):** Splitting features into low-frequency approximations and high-frequency details (horizontal, vertical, diagonal).
- **SpatialMamba:** 2D multi-direction selective scan state space model processing the low-frequency representation with $O(N)$ linear complexity.
- **StripConvStack:** Directional strip convolutions enhancing high-frequency linear spatial structures (roads, field boundaries).
- **Progressive Upsampler:** 2× + 2× progressive `PixelShuffle` stages, each followed by `NAFBlock` (Non-linear Activation Free) refinement.
- **UQ Heads:** Dual projection heads outputting mean $\mu$ (12-band surface reflectance) and log-variance $\log(\sigma^2)$ (aleatoric uncertainty).

### 3. Teacher Distillation Setup
- **Teacher Model:** Pretrained `OpenSR-SRGAN` (`simon-donike/SR-GAN`), ~1.50M parameters, supervising 4 bands: B2, B3, B4, B8 (RGB-NIR).
- **Baseline Training (Stage 2/3):** Trained for 8,000 steps using standard $L_1$ Teacher Loss + MTF Sensor Degradation Loss on 12 bands + Spectral Angle Mapper (SAM).
- **Baseline Convergence:** Reached best Validation Teacher L1 of **0.0075** (Stage 2) and **0.0069** (Stage 3).

### 4. Diagnosis of the Blurriness Phenomenon
- Visual inspection of the 2.5m outputs revealed that despite strong metrics, outputs were visibly oversmoothed compared to the sharp SRGAN teacher target.
- Root Cause Identified: $L_1$ loss mathematically penalizes spatial variance and rewards the median blur of plausible textures.

### 5. Decisive 1-Patch Diagnostic Test
- To prove whether the 1.11M Mamba architecture was mathematically capable of reproducing sharp high-frequency edges, we ran an isolated overfitting test on a single patch from scratch:
  - **Script:** `scripts/train_one_patch.py`
  - **Losses:** $L_1$ replaced with **Charbonnier Loss** + **Image Gradient Loss** ($L_1(dx) + L_1(dy)$). MTF and SAM disabled to isolate edge learning.
  - **Result:** In 1,500 steps on a Modal T4 GPU (106 seconds), the model reached **47.27 dB PSNR**, driving residual error to near-zero and proving that the 1.11M architecture has ample capacity for razor-sharp edges.

### 6. Stage 4 Edge-Aware Fine-Tuning Run
- Based on the diagnostic test, we updated the main distillation training loop (`scripts/distill_srgan.py`):
  - **Checkpoint Preservation:** Warm-started from the existing converged weights; did not alter layer dimensions or tensor shapes.
  - **Optimizer Reset & Warmup:** Reset AdamW optimizer state to purge old L1 momentum and configured `OneCycleLR` with a 10% warmup (starting from $10^{-6}$ up to $5 \times 10^{-5}$) to eliminate gradient shock.
  - **Loss Formulation:**
    $$\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{Charbonnier}}(\text{RGBN}) + 0.20 \times \mathcal{L}_{\text{Gradient}}(\text{RGBN}) + 0.05 \times \mathcal{L}_{\text{MTF}}(12\text{b}) + 0.01 \times \mathcal{L}_{\text{SAM}}(\text{RGBN})$$
    *(Weak MTF and SAM retained as guardrails to prevent SWIR band degradation and color shifts).*
  - **Throughput:** Increased batch size to 32, maxing out the 24GB A10G VRAM (~18–20 GB usage).
  - **Execution:** Ran for 8,000 steps on Modal NVIDIA A10G.

---

## Part 3: Where We Are Right Now (Current State & Evidence)

### 1. Training Loss Benchmark
- **Previous Best Validation L1:** `0.0075` (Stage 2) $\rightarrow$ `0.0069` (Stage 3)
- **Current Edge Fine-tuned Best Validation L1:** **`0.0064`** (saved at `checkpoints/best.pt`)
- **Net Improvement:** **15% lower validation error** than the original converged baseline.

### 2. Comprehensive Test Metrics (Evaluated on 50 Completely Unseen Test Patches)
Evaluated on held-out test scenes from a separate acquisition date that were never seen during training or validation:

| Metric | Bicubic 4× Baseline (2.5m) | **TriSpecWave Student (2.5m)** | OpenSR-SRGAN Teacher (2.5m) | Delta vs. Baseline |
|---|---|---|---|---|
| **PSNR** | 30.04 dB | **32.17 dB** | *Reference Target* | 🟢 **+2.13 dB** |
| **SSIM** | 0.8016 | **0.8754** | *Reference Target* | 🟢 **+0.0738** |
| **SAM (Color Angle)** | 0.0242 rad | **0.0198 rad** (~1.13°) | *Reference Target* | 🟢 **-18.2% error** |
| **NDVI MAE** | 0.0226 | **0.0161** | *Reference Target* | 🟢 **-28.7% error** |
| **Inference Latency** | ~1.5 ms | **24.96 ms** | 47.03 ms | 🟢 **1.88× faster** |
| **Parameter Count** | 0 | **1.117M** | ~1.50M | 🟢 **25% smaller** |
| **Inference VRAM** | <50 MB | **~270 MB** | ~1.2 GB | 🟢 **Ultra-light** |
| **Bands Super-Resolved** | 12 bands | **12 bands** | 4 bands only | 🟢 **Full spectrum** |

### 3. Per-Band Absolute Errors (Student vs. Teacher)
- Band 2 (Blue): `0.00739`
- Band 3 (Green): `0.00766`
- Band 4 (Red): `0.00863`
- Band 8 (NIR): `0.00726`

### 4. Visual Inspection Status
Located in `outputs/`:
- `unseen_patch_1.png` through `unseen_patch_4.png`: Side-by-side comparisons of True Color (RGB), False Color (NIR), Absolute Error Heatmaps, and Predicted Aleatoric Uncertainty.
- `pune_comparison.png`: 6-patch multi-row comparison grid across diverse terrains (urban, agricultural, water bodies, linear roads).
- **Current Visual Status:** Edges, field boundaries, and road networks are visibly sharper than Bicubic baseline and noticeably crisper than the Stage 2/3 L1 model. However, fine stochastic textures (e.g. dense crop speckle) are still slightly smoother than the GAN teacher's pseudo-target.

---

## Part 4: File & Directory Manifest

```
.
├── AI_ANALYSIS_HANDOFF.md         <-- This comprehensive document
├── configs/
│   └── distill_pune.yaml          <-- Active hyperparameter and band definition config
├── models/
│   ├── trispecwave_mamba_uq.py    <-- Top-level 12-band student architecture
│   ├── spatial_mamba.py           <-- 2D multi-direction selective scan state space
│   ├── spectral_mamba.py          <-- 1D spectral state space with nm/GSD embeddings
│   ├── progressive_upsampler.py   <-- 2x+2x progressive PixelShuffle with NAF blocks
│   ├── wavelet_transform.py       <-- 2D Haar DWT / IDWT decomposition
│   ├── naf_blocks.py              <-- Non-linear Activation Free convolution blocks
│   ├── strip_conv.py              <-- Directional strip convolution filters
│   ├── gated_fusion.py            <-- Cross-resolution gated fusion
│   ├── graph_fusion.py            <-- Spectral graph mixer
│   └── uq_heads.py                <-- Heteroscedastic Gaussian uncertainty head
├── losses/
│   ├── mtf_psf_loss.py            <-- Physical satellite sensor degradation MTF/PSF
│   ├── spectral_consistency_loss.py<-- Spectral Angle Mapper (SAM)
│   └── heteroscedastic_loss.py    <-- Gaussian NLL uncertainty loss
├── metrics/
│   ├── spatial_metrics.py         <-- PSNR & SSIM computation
│   └── spectral_metrics.py        <-- Spectral angle metrics
├── scripts/
│   ├── distill_srgan.py           <-- Main distillation training pipeline (Charbonnier + Grad)
│   ├── train_one_patch.py         <-- Diagnostic 1-patch overfitting script
│   ├── evaluate_distillation.py   <-- Comprehensive evaluation & figure generation suite
│   ├── prepare_pune_patches.py    <-- Sentinel-2 SCL cloud masking & tile extractor
│   └── download_pune_sentinel2.py <-- STAC API Sentinel-2 search & download
├── modal_app.py                   <-- Serverless cloud orchestration (Modal GPU functions)
├── checkpoints/
│   ├── best.pt                    <-- Latest converged checkpoint weights (Val L1 = 0.0064)
│   └── edge_finetune_best.pt      <-- Explicit copy of edge-finetuned checkpoint
└── outputs/
    ├── metrics.json               <-- Quantitative evaluation log
    ├── pune_comparison.png        <-- 6-patch unseen test comparison grid
    ├── unseen_patch_1.png         <-- High-res individual test comparison 1
    ├── unseen_patch_2.png         <-- High-res individual test comparison 2
    ├── unseen_patch_3.png         <-- High-res individual test comparison 3
    ├── unseen_patch_4.png         <-- High-res individual test comparison 4
    ├── one_patch_comparison.png   <-- 47.27 dB 1-patch diagnostic test visualization
    ├── pune_student_rgbn.png      <-- True Color (RGB) vs False Color (NIR) preview
    └── pune_uncertainty.png       <-- Predicted aleatoric uncertainty maps
```
