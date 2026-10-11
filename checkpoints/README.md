# Model Checkpoints & Architecture Specs

This directory houses the trained weights for **TriSpecWave-Mamba-UQ**, a 12-band multi-spectral super-resolution model for Sentinel-2 Level-2A imagery.

---

## 1. Checkpoint Inventory

| Checkpoint File | Parameter Count | File Size | Architecture / Role | Training Dataset | Validation Loss |
| :--- | :---: | :---: | :--- | :--- | :---: |
| `sen2neon_6m_best.pt` | **6,172,314** | **24.8 MB** | `TriSpecWaveMambaUQ` Base 12-band super-resolution model | ESA OpenSR SEN2NEON (32,672 patches) | **0.03095** |
| `sen2neon_unet_adapter_epoch_3.pt` | **~230,000** | **914 KB** | `RGBNUNetAdapter` High-frequency edge detailer (B2, B3, B4, B8) | SEN2NEON Phase 1 Cache | N/A |

---

## 2. Model Architecture Parameters (`sen2neon_6m.yaml`)

```yaml
in_channels: 12
out_channels: 12
feature_dim: 128
spectral_dim: 64
naf_depth: 8
spatial_depth: 4
graph_heads: 4
wavelet_levels: 1
upsample_stages: 2
scale: 4
dropout: 0.05

band_wavelengths_nm: [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
band_gsd_m: [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 20, 20]
```

---

## 3. Training & Validation Splits
- **Source**: ESA OpenSR SEN2NEON dataset.
- **Train Split**: 32,672 patches ($64 \times 64$ LR $\rightarrow$ $256 \times 256$ HR), 12 bands.
- **Validation Split**: 3,632 patches (strictly held out).
- **Loss**: Shifted Window Total Loss ($\pm 1$ pixel spatial jitter tolerance) + Sobel Image Gradient Loss + Dynamic Nodata border masking.
- **Optimizer**: AdamW with differential learning rates ($10^{-5}$ feature backbone, $10^{-4}$ upsampler/tail) and cosine annealing.
