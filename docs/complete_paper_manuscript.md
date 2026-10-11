# TriSpecWaveMambaUQ: Physics-Informed Multi-Spectral Super-Resolution with State Space Models and Calibrated Uncertainty Quantification for Sentinel-2 MSI

**Khushalzz et al.**  
*Laboratory for Geospatial Intelligence and Earth Observation Machine Learning*  
*IEEE Transactions on Geoscience and Remote Sensing (TGRS) / Technical Manuscript*

---

### Abstract
Multispectral satellite imagery from the Sentinel-2 Multi-Spectral Instrument (MSI) is physically constrained by optical detector geometry, distributing twelve reflective spectral bands across three disparate spatial resolutions ($10\,\text{m}$, $20\,\text{m}$, and $60\,\text{m}$). Conventional single-image super-resolution (SISR) methods struggle with this multi-resolution data: Convolutional Neural Networks (CNNs) have small receptive fields, Vision Transformers (ViTs) incur prohibitive quadratic computational complexity $\mathcal{O}(N^2)$ causing GPU memory explosion on large satellite tiles, and standard generative models (GANs) hallucinate non-existent ground features without predictive confidence bounds.

In this work, we propose **TriSpecWaveMambaUQ**, an end-to-end deep neural architecture ($6.17\,\text{M}$ parameters, $1.12\,\text{M}$ base backbone) for $4\times$ multispectral super-resolution ($10\,\text{m} \rightarrow 2.5\,\text{m}$ Ground Sampling Distance) with calibrated pixel-wise uncertainty quantification. Our model processes all 12 bands via a tri-stream grouped encoder aligned with native sensor resolutions, dynamically captures inter-band correlations via 3-node Spectral Graph Attention and continuous-wavelength 1D State Space Models (SpectralMamba, $\mathcal{O}(C)$), and losslessly decouples spatial features through 2D Haar Discrete Wavelet Transforms. Low-frequency background structures are routed through linear-complexity 2D SpatialMamba blocks ($\mathcal{O}(N)$), while high-frequency details pass through directional Strip Convolution stacks ($1\times7$ and $7\times1$). Dual output heads simultaneously reconstruct surface reflectance means ($\boldsymbol{\mu}$) and aleatoric log-variance uncertainty maps ($\log \boldsymbol{\sigma}^2$) under a heteroscedastic Gaussian negative log-likelihood objective regularized by physical sensor Modulation Transfer Function (MTF) degradation constraints.

Evaluated on unseen geographic test suites against Bicubic interpolation and the OpenSR-SRGAN teacher, our model achieves **$32.17\,\text{dB}$ PSNR** ($+2.13\,\text{dB}$ gain), **$0.8754$ SSIM**, an **$18.2\%$ reduction in Spectral Angle Mapper error ($0.0198\,\text{rad}$ / $1.13^\circ$)**, and a **$28.7\%$ reduction in NDVI MAE ($0.0161$)**, with an inference latency of **$24.96\,\text{ms}$** per tile and **$270\,\text{MB}$ peak VRAM**. Calibrated uncertainty maps reliably flag ambiguous parcel boundaries and cloud fringes, establishing an operational foundation for reliable Earth observation.

**Keywords:** Sentinel-2 MSI, Multispectral Super-Resolution, State Space Models (SSM), Mamba, Haar Discrete Wavelet Transform, Uncertainty Quantification, Modulation Transfer Function (MTF).

---

## I. Introduction

Satellite Earth observation provides the sensory foundation for climate monitoring, food security forecasting, and disaster mitigation. The Multi-Spectral Instrument (MSI) aboard the European Space Agency's (ESA) Copernicus Sentinel-2 constellation is an operational cornerstone in open-access Earth observation. MSI acquires 12 distinct reflective spectral bands spanning from the visible and near-infrared (VNIR) to the short-wave infrared (SWIR) spectrum. 

However, optical satellite payload design involves fundamental physical trade-offs between orbital swath width, Signal-to-Noise Ratio (SNR), and optical detector dimensions. Consequently, Sentinel-2 distributes its 12 reflective bands across three disparate spatial Ground Sampling Distances (GSD):
- **$10\,\text{m}$ High-Resolution Bands:** Band 2 (Blue, $490\,\text{nm}$), Band 3 (Green, $560\,\text{nm}$), Band 4 (Red, $665\,\text{nm}$), and Band 8 (Broad NIR, $842\,\text{nm}$).
- **$20\,\text{m}$ Medium-Resolution Bands:** Band 5 (Red Edge 1, $705\,\text{nm}$), Band 6 (Red Edge 2, $740\,\text{nm}$), Band 7 (Red Edge 3, $783\,\text{nm}$), Band 8A (Narrow NIR, $865\,\text{nm}$), Band 11 (SWIR-1, $1610\,\text{nm}$), and Band 12 (SWIR-2, $2190\,\text{nm}$).
- **$60\,\text{m}$ Coarse Atmospheric Bands:** Band 1 (Coastal Aerosol, $443\,\text{nm}$) and Band 9 (Water Vapor, $945\,\text{nm}$).

Super-resolving all twelve bands to a uniform sub-$10\,\text{m}$ resolution (specifically, a $4\times$ upscaling to $2.5\,\text{m}$ GSD) is essential for precision agriculture, wildfire burn-scar severity mapping, urban infrastructure monitoring, and environmental hazard tracking. Despite substantial progress in Single-Image Super-Resolution (SISR), four fundamental engineering bottlenecks persist:

1. **Native Sensor Resolution Heterogeneity:** Standard SISR models assume uniform 3-channel RGB inputs. Applying them to Sentinel-2 typically requires naively upsampling coarse $20\,\text{m}$ and $60\,\text{m}$ bands prior to feature extraction, smearing subtle spectral signatures and distorting downstream vegetation and moisture indices.
2. **The Receptive Field vs. Complexity Dilemma:** Convolutional Neural Networks (CNNs) rely on small localized filters that miss large-scale geographical context. Conversely, Vision Transformers (ViTs) compute all-to-all attention with quadratic complexity $\mathcal{O}(N^2)$ relative to pixel count. On large satellite orthomosaics with millions of pixels, ViTs suffer GPU Out-Of-Memory (OOM) crashes and introduce blocky window tiling artifacts.
3. **The Hallucination vs. Blur Trade-Off:** Optimizing models under standard pixel regression ($L_1$ or $L_2$) produces overly smooth, blurry outputs by predicting the conditional mean $\mathbb{E}[\mathbf{y} \mid \mathbf{x}]$. Unconstrained Generative Adversarial Networks (GANs) synthesize sharp high frequencies, but frequently hallucinate non-existent physical features (e.g., drawing fake roads or buildings) that corrupt scientific analysis.
4. **The Operational Reliability Void:** Automated downstream pipelines (e.g. crop insurance auditing) require calibrated confidence bounds. Current super-resolution models operate as deterministic black boxes with zero awareness of their own predictive uncertainty in complex terrain, cloud edges, or shadows.

### Core Contributions
To resolve these bottlenecks, we introduce **TriSpecWaveMambaUQ**, an end-to-end physics-informed framework for $4\times$ multispectral super-resolution ($10\,\text{m} \rightarrow 2.5\,\text{m}$ GSD) across all 12 Sentinel-2 MSI bands:
- **Tri-Stream Grouped Encoder:** Ingests bands at their native physical resolutions ($10\,\text{m}, 20\,\text{m}, 60\,\text{m}$) and fuses them via a learned Gated Bilinear Upsampler, avoiding destructive pre-interpolation.
- **Continuous-Wavelength State Space Modeling:** Pairs a 3-node Spectral Graph Attention network with 1D `SpectralMamba` state-space blocks conditioned on central physical wavelengths $\lambda_b \in [443, 2190]\,\text{nm}$, achieving global cross-band context in linear $\mathcal{O}(C)$ complexity.
- **Frequency-Decoupled Spatial Processing:** Leverages a 2D Haar Discrete Wavelet Transform (`HaarDWT2D`) to route low-frequency background features through a linear-complexity 2D `SpatialMamba` ($\mathcal{O}(N)$) and high-frequency edge details through directional `StripConv` stacks ($1\times7$ and $7\times1$).
- **Physics-Informed Sensor MTF Loss:** Enforces a closed-loop optical Point Spread Function / Modulation Transfer Function (MTF) degradation constraint, mathematically guaranteeing that downscaling our output recreates the original raw satellite observation.
- **Calibrated Uncertainty Quantification:** Simultaneously predicts the surface reflectance mean $\boldsymbol{\mu}$ and pixel-wise aleatoric log-variance $\log \boldsymbol{\sigma}^2$ under a heteroscedastic Gaussian Negative Log-Likelihood objective.
- **Pareto-Optimal Edge Efficiency:** Achieves state-of-the-art results (**$32.17\,\text{dB}$ PSNR**, **$0.8754$ SSIM**, **$0.0198\,\text{rad}$ SAM**) with a compact **$6.17\,\text{M}$ parameter footprint** ($1.12\,\text{M}$ base backbone), running in **$24.96\,\text{ms}$** with **$270\,\text{MB}$ peak VRAM**.

---

## II. Related Work

Single-Image Super-Resolution (SISR) has evolved from early residual convolutional networks to generative adversarial models, vision transformers, and selective state-space architectures. Table I summarizes key milestones.

### Table I: Chronological Comparison of Super-Resolution Paradigms

| Model | Year | Paradigm | Bands | Spatial Complexity | Physics / MTF | Uncertainty | Primary Bottleneck in Satellite SR |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **SRCNN** | 2014 | CNN | 3 | $\mathcal{O}(K^2 N)$ | No | No | Tiny receptive field; pre-upsampling compute overhead |
| **VDSR** | 2016 | Residual CNN | 3 | $\mathcal{O}(L \cdot K^2 N)$ | No | No | Deep isotropic convolutions oversmooth agricultural boundaries |
| **EDSR** | 2017 | Residual CNN | 3 | $\mathcal{O}(B \cdot K^2 N)$ | No | No | Parameter heavy (~43M); ignores native multi-resolution bands |
| **SRGAN** | 2017 | GAN | 3 | $\mathcal{O}(K^2 N)$ | No | No | Hallucinates non-existent ground textures; corrupts radiometry |
| **DSen2** | 2018 | RS CNN | 12 | $\mathcal{O}(K^2 N)$ | Partial | No | Restricted to 10m target; isotropic $L_1$ blur; purely deterministic |
| **RCAN** | 2018 | Channel Attn | 3 | $\mathcal{O}(K^2 N) + \mathcal{O}(C^2)$ | No | No | Abstract channel attention ignores continuous physical wavelengths |
| **SwinIR** | 2021 | Transformer | 3 | $\mathcal{O}(M^2 N)$ | No | No | Shifted-window tiling artifacts; heavy memory footprint |
| **Restormer** | 2022 | Transformer | 3 | $\mathcal{O}(C^2 N)$ | No | No | Transposed channel attention computationally prohibitive for 12 bands |
| **OpenSR-SRGAN** | 2023 | RS GAN | 4 | $\mathcal{O}(K^2 N)$ | Empirical | No | Restricted to 4 bands (RGB-NIR); hallucinates spectral signatures |
| **MambaIR** | 2024 | SSM | 3 | $\mathcal{O}(N)$ Linear | No | No | RGB-only; isotropic scanning ignores multi-band physical spectra |
| **TriSpecWaveMambaUQ** | *Ours* | **Hybrid SSM+UQ** | **Full 12** | **$\mathcal{O}(N)$ Linear** | **Yes** | **Yes** | **Unified 12-band $4\times$ downscaling to 2.5m; calibrated UQ; 24.96 ms** |

### Key Architectural Takeaways for CSE Students
1. **From Convolutions to State Spaces:** CNNs have small receptive fields, and Transformers incur quadratic complexity $\mathcal{O}(N^2)$. State Space Models (Mamba) behave like recurrent neural networks with input-dependent weights that can be parallelized during training via prefix scans, running in **strictly linear time $\mathcal{O}(N)$**.
2. **Frequency Decoupling:** Wavelet transforms cleanly split an image into low frequencies (broad illumination and land cover) and high frequencies (sharp edges and boundaries), allowing specialized neural processing for each.
3. **Physics-Guided Guardrails:** Without physical constraints, neural networks invent plausible-looking fake textures. Enforcing sensor optical blur (MTF) consistency acts as an autoencoder sanity check that prevents hallucinations.
4. **Aleatoric Uncertainty Modeling:** Formulating super-resolution under a heteroscedastic Gaussian likelihood forces the model to predict both an answer (mean) and a confidence score (variance), giving automated systems an interpretable safety metric.

---

## III. Methodology

### A. Architectural Overview
TriSpecWaveMambaUQ super-resolves all 12 Sentinel-2 bands from their native resolutions ($10\,\text{m}, 20\,\text{m}, 60\,\text{m}$) to a uniform $2.5\,\text{m}$ Ground Sampling Distance ($4\times$ scale factor).

```
========================================================================================
                          TriSpecWaveMambaUQ ARCHITECTURE
========================================================================================
Input: 12-Band Sentinel-2 MSI Cube (Native Resolutions)
  ├── 10m Group: B2, B3, B4, B8        ──► GroupConv10m (4 -> C) ─────────────┐
  ├── 20m Group: B5..B7, B8A, B11, B12 ──► GroupConv20m (6 -> C) ──► Bilinear 2x ┤
  └── 60m Group: B1, B9                ──► GroupConv60m (2 -> C) ──► Bilinear 6x ┘
                                                                               │
                                    ┌──────────────────────────────────────────┘
                                    ▼
                         [Gated Bilinear Fusion]
                                    │
                    [3-Node Spectral Graph Attention]  (Cross-GSD Topological Routing)
                                    │
                    [1D Continuous SpectralMamba]     (Wavelength-Aware SSM, O(C))
                                    │
                    [2D Haar Wavelet Decomposition]
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
         Low-Frequency (LL)               High-Frequency (LH, HL, HH)
         [2D SpatialMamba]                [Directional StripConvs (1x7, 7x1)]
         (Global Receptive Field, O(N))   [NAF Residual Feature Blocks]
                   └────────────────┬────────────────┘
                                    │
                    [Inverse Haar Wavelet Synthesis]
                                    │
                  [Progressive 4x PixelShuffle Upsampling]
                             (2x -> Conv -> 2x)
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
        [Predictive Mean Head μ]           [Aleatoric Log-Variance Head log σ²]
        12 Bands @ 2.5m GSD                12 Bands Calibrated Pixel Variance
                   │                                 │
                   └────────────────┬────────────────┘
                                    ▼
                        [Physics-Informed Training]
         Heteroscedastic NLL  +  Sensor MTF  +  Shifted-Window Edge  +  SAM
========================================================================================
```

### B. Tri-Stream Native Feature Extraction and Gated Fusion
Sentinel-2 bands are grouped by their native detector pixel pitch:
$$\mathcal{I}_{10\text{m}} = \{2, 3, 4, 8\}, \quad \mathcal{I}_{20\text{m}} = \{5, 6, 7, 8\text{A}, 11, 12\}, \quad \mathcal{I}_{60\text{m}} = \{1, 9\}$$

Each group is passed to a dedicated grouped convolutional encoder $\mathcal{E}_g$:
$$\mathbf{F}_{10\text{m}} = \mathcal{E}_{10\text{m}}(\mathbf{X}_{10\text{m}}), \quad \mathbf{F}_{20\text{m}} = \mathcal{E}_{20\text{m}}(\mathbf{X}_{20\text{m}}), \quad \mathbf{F}_{60\text{m}} = \mathcal{E}_{60\text{m}}(\mathbf{X}_{60\text{m}})$$

To reconcile spatial dimensions without destroying raw spectral integrity, features $\mathbf{F}_{20\text{m}}$ and $\mathbf{F}_{60\text{m}}$ are upsampled via bilinear interpolation to match the $10\,\text{m}$ grid and merged via learned gating:
$$\mathbf{F}_{\text{fused}} = \mathbf{F}_{10\text{m}} + \mathbf{W}_{\text{gate}} \ast \left[ \mathbf{F}_{20\text{m}}^\uparrow, \, \mathbf{F}_{60\text{m}}^\uparrow \right]$$
*Intuition for CSE Students:* Instead of blindly resizing images in preprocessing, we let the network learn how much coarse information to inject into the sharp $10\,\text{m}$ grid using a convolutional gate.

### C. Spectral Dependency Modeling: Spectral Graph + SpectralMamba
The 12 channels represent physical electromagnetic wavelengths from $443\,\text{nm}$ to $2190\,\text{nm}$. To capture radiative correlations without quadratic compute overhead:
1. **3-Node Spectral Graph Attention:** Treats the three resolution streams as nodes in a graph, computing scaled dot-product attention to route global information between resolution groups.
2. **Continuous-Wavelength SpectralMamba:** Each channel is mapped to a token conditioned on its central physical wavelength $\lambda_b \in [443, 2190]\,\text{nm}$ and native GSD. The sequence is processed by a 1D selective state-space scan:
   $$h_b = \bar{\mathbf{A}}_b h_{b-1} + \bar{\mathbf{B}}_b x_b, \quad y_b = \mathbf{C}_b h_b$$
   *Intuition for CSE Students:* In standard RNNs, weight matrices $\mathbf{A}, \mathbf{B}$ are static constants. In Mamba, $\bar{\mathbf{A}}_b, \bar{\mathbf{B}}_b$ are dynamically calculated from the input token itself. This gives Mamba the reasoning power of attention while running in **strictly linear time $\mathcal{O}(C)$**.

### D. Frequency-Decoupled Spatial Processing
Spatial feature extraction is separated into frequency regimes:
1. **2D Haar Wavelet Decomposition:** Splits spatial features into four orthogonal subbands:
   $$\{LL, LH, HL, HH\} = \text{HaarDWT2D}(\mathbf{F}_{\text{spec}})$$
   where $LL$ contains smooth structural energy, and $LH, HL, HH$ contain horizontal, vertical, and diagonal edges.
2. **Low-Frequency Routing $\rightarrow$ 2D SpatialMamba:** The $LL$ subband is processed via a 2D multi-direction selective scan block. By scanning across horizontal and vertical raster paths, `SpatialMamba` provides a global spatial receptive field with strictly linear computational complexity $\mathcal{O}(N)$, avoiding the $\mathcal{O}(N^2)$ memory explosion of Transformers.
3. **High-Frequency Routing $\rightarrow$ Directional Strip Convolutions:** The edge subbands $(LH, HL, HH)$ are processed by asymmetric strip convolutions ($1\times7$ and $7\times1$ kernels) and Non-linear Activation Free (`NAF`) residual blocks, explicitly tracing linear features like roads, field edges, and canals.
4. **Wavelet Synthesis:** The subbands are recombined via the Inverse Haar Discrete Wavelet Transform (`IDWT2D`), losslessly reconstructing the spatial feature map.

### E. Dual Reconstruction and Uncertainty Heads
Synthesized features are upscaled to $2.5\,\text{m}$ resolution through two progressive $2\times$ PixelShuffle stages. The network branches into dual output heads:
- **Predictive Mean Head ($\boldsymbol{\mu}$):** Reconstructs the 12-band super-resolved surface reflectance:
  $$\boldsymbol{\mu} = \mathbf{W}_\mu \ast \mathbf{F}_{\text{up}} + \mathbf{X}_{\text{bilinear}}^\uparrow$$
  where a global bilinear skip connection preserves baseline low-frequency radiometry.
- **Aleatoric Uncertainty Head ($\log \boldsymbol{\sigma}^2$):** Predicts the per-pixel log-variance $\mathbf{s} = \log \boldsymbol{\sigma}^2$, clamped to $[-10, 5]$ for numerical stability:
  $$\mathbf{s} = \text{clamp}(\mathbf{W}_\sigma \ast \mathbf{F}_{\text{up}}, -10, 5)$$

### F. Physics-Informed Training Objectives
The network is optimized end-to-end under four complementary loss functions:

1. **Heteroscedastic Gaussian Negative Log-Likelihood Loss:**
   $$\mathcal{L}_{\text{hetero}} = \frac{1}{2 B \Omega} \sum_{b=1}^{B} \sum_{i \in \Omega} \left( \exp(-s_{b,i}) \vert y_{b,i} - \mu_{b,i} \vert + s_{b,i} \right)$$
   *Intuition for CSE Students:* If an edge or cloud boundary is inherently ambiguous, the model increases variance $s_{b,i}$ to soften the error penalty. Meanwhile, the second term $+s_{b,i}$ prevents the model from cheating by outputting infinite uncertainty everywhere.
2. **Sensor MTF (Optical Degradation) Loss:**
   $$\mathcal{L}_{\text{MTF}} = \frac{1}{B} \sum_{b=1}^{B} \Vert \mathcal{D}_4(\mathbf{k}_{\text{PSF}} \ast \boldsymbol{\mu}_b) - \mathbf{x}_{b,\text{LR}} \Vert_1$$
   *Physics Cycle-Consistency Check:* If we blur our super-resolved $2.5\,\text{m}$ output with the camera's optical Point Spread Function ($\mathbf{k}_{\text{PSF}}$, Gaussian $\sigma=1.2$) and downsample it back to native resolution, it **must match the raw satellite observation**. This prevents the model from hallucinating fake terrain features.
3. **Shifted-Window Edge Loss ($\mathcal{L}_{\text{SW}}$):** Accommodates sub-pixel sensor co-registration offsets ($\pm 1\,\text{pixel}$) using a localized minimum-residual search combined with Sobel gradient matching ($\nabla_x, \nabla_y$).
4. **Spectral Angle Mapper (SAM) Loss:** Enforces angular vector alignment across all 12 bands to eliminate color and spectral distortion:
   $$\mathcal{L}_{\text{SAM}} = \frac{1}{\Omega} \sum_{i \in \Omega} \arccos\left( \frac{\boldsymbol{\mu}_i^\top \mathbf{y}_i}{\Vert \boldsymbol{\mu}_i \Vert_2 \, \Vert \mathbf{y}_i \Vert_2 + \epsilon} \right)$$
5. **Total Training Objective:**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{hetero}} + \lambda_{\text{MTF}} \mathcal{L}_{\text{MTF}} + \lambda_{\text{SW}} \mathcal{L}_{\text{SW}} + \lambda_{\text{SAM}} \mathcal{L}_{\text{SAM}}$$
   configured with weights $\lambda_{\text{MTF}} = 0.1$, $\lambda_{\text{SW}} = 1.0$, and $\lambda_{\text{SAM}} = 0.05$.

---

## IV. Experiments & Results

### A. Datasets & Geographic Benchmark Regions
We evaluate TriSpecWaveMambaUQ on two complementary datasets spanning diverse agro-urban and high-altitude terrains:
1. **Pune Metropolitan Region AOI (Maharashtra, India):**
   - **Footprint:** Centered at $18.4388^\circ\,\text{N}, 73.7728^\circ\,\text{E}$ ($10\,\text{km}$ radius). Captures dense urban settlements, fragmented smallholder agricultural plots, the Khadakwasla reservoir, and transport corridors.
   - **Data Characteristics:** Cloud-free Sentinel-2 Level-2A Bottom-Of-Atmosphere (BOA) surface reflectance tiles (2023–2026, cloud cover $\le 2\%$).
   - **Splits:** Sliced into $64 \times 64$ LR patches ($10\,\text{m}$ GSD) mapping to $256 \times 256$ HR targets ($2.5\,\text{m}$ GSD). Chronologically split into **1,400 training**, **100 validation**, and **50 held-out unseen test patches**.
2. **ESA OpenSR SEN2NEON Reference Benchmark:**
   - Pairs observed Sentinel-2 L2A BOA reflectances with co-registered $1\,\text{m}$ AVIRIS-NG airborne hyperspectral data harmonized to Sentinel-2 Spectral Response Functions on a standardized $2.5\,\text{m}$ HR grid (32,672 training, 3,632 test patches).

### B. Implementation, Hardware & Training Pipeline
- **Hardware:** Trained on an NVIDIA A10G Tensor Core GPU ($24\,\text{GB}$ VRAM) with scale verification on an NVIDIA B200.
- **In-Memory Caching:** Pinned-host RAM caching (`TeacherCacheDataset` and `sen2neon_ram_dataset`) bypassed remote storage I/O bottlenecks, increasing training throughput by $10.6\times$ ($0.3 \rightarrow 3.2\,\text{steps/s}$).
- **Optimization:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$), initial learning rate $\eta_0 = 2 \times 10^{-4}$, Cosine Annealing decay down to $\eta_{\text{min}} = 10^{-6}$ over 50 epochs.
- **Precision & Batching:** Mixed-precision training via PyTorch AMP (`torch.amp.autocast(dtype=torch.bfloat16)`) with effective batch size of 16 patches.
- **Distillation:** The base student backbone was distilled from OpenSR-SRGAN on the 4 RGB-NIR bands while all 12 bands were jointly optimized under MTF degradation and heteroscedastic NLL objectives.
- **Inference Runtime:** Exported to ONNX and TensorRT engine formats, running in $24.96\,\text{ms}$ on GPU and $<120\,\text{ms}$ on multi-core CPUs.

### C. Quantitative Evaluation Metrics
1. **Peak Signal-to-Noise Ratio (PSNR):**
   $$\text{PSNR}(\mathbf{y}, \hat{\mathbf{y}}) = 10 \cdot \log_{10}\left( \frac{\text{MAX}_I^2}{\text{MSE}(\mathbf{y}, \hat{\mathbf{y}})} \right)$$
   where $\text{MAX}_I = 1.0$ for normalized reflectance $\rho \in [0, 1]$. Higher is better.
2. **Structural Similarity Index Measure (SSIM):**
   $$\text{SSIM}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{(2\mu_{\mathbf{y}}\mu_{\hat{\mathbf{y}}} + c_1)(2\sigma_{\mathbf{y}\hat{\mathbf{y}}} + c_2)}{(\mu_{\mathbf{y}}^2 + \mu_{\hat{\mathbf{y}}}^2 + c_1)(\sigma_{\mathbf{y}}^2 + \sigma_{\hat{\mathbf{y}}}^2 + c_2)}$$
   Measures perceptual structural alignment. Higher is better.
3. **Spectral Angle Mapper (SAM):**
   Measures pixel-wise spectral vector distortion in radians and degrees:
   $$\text{SAM}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{\Omega} \sum_{i \in \Omega} \arccos\left( \frac{\mathbf{y}_i^\top \hat{\mathbf{y}}_i}{\Vert \mathbf{y}_i \Vert_2 \, \Vert \hat{\mathbf{y}}_i \Vert_2 + \epsilon} \right)$$
   Lower indicates superior spectral and color conservation.
4. **Biophysical Vegetation & Moisture Index Errors:**
   - **Normalized Difference Vegetation Index (NDVI):**
     $$\text{NDVI} = \frac{\rho_{\text{B8}} - \rho_{\text{B4}}}{\rho_{\text{B8}} + \rho_{\text{B4}}}, \quad \text{MAE}_{\text{NDVI}} = \frac{1}{\Omega}\sum_{i \in \Omega} \vert \text{NDVI}(\mathbf{y}_i) - \text{NDVI}(\hat{\mathbf{y}}_i) \vert$$
   - **Normalized Difference Red Edge Index (NDRE):**
     $$\text{NDRE} = \frac{\rho_{\text{B8}} - \rho_{\text{B5}}}{\rho_{\text{B8}} + \rho_{\text{B5}}}, \quad \text{MAE}_{\text{NDRE}} = \frac{1}{\Omega}\sum_{i \in \Omega} \vert \text{NDRE}(\mathbf{y}_i) - \text{NDRE}(\hat{\mathbf{y}}_i) \vert$$
   - **Normalized Difference Water Index (NDWI):**
     $$\text{NDWI} = \frac{\rho_{\text{B3}} - \rho_{\text{B8}}}{\rho_{\text{B3}} + \rho_{\text{B8}}}, \quad \text{MAE}_{\text{NDWI}} = \frac{1}{\Omega}\sum_{i \in \Omega} \vert \text{NDWI}(\mathbf{y}_i) - \text{NDWI}(\hat{\mathbf{y}}_i) \vert$$
5. **Uncertainty Calibration:** Quantified via the Spearman rank correlation $\rho(\boldsymbol{\sigma}^2, \vert \mathbf{y} - \boldsymbol{\mu} \vert)$ between predicted variance and true reconstruction absolute error.

### D. Quantitative Performance Benchmark

### Table II: Overall Performance Benchmark (Pune Unseen Test Suite, 50 Patches, 10m -> 2.5m)

| Model | Parameters | Latency (ms) | Peak VRAM | Bands | PSNR (dB) ↑ | SSIM ↑ | SAM (rad) ↓ | SAM (deg) ↓ | $\text{MAE}_{\text{NDVI}}$ ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | — | — | — | 12 | 30.04 | 0.8016 | 0.0242 | 1.39° | 0.0226 |
| **OpenSR-SRGAN (Teacher)** | 1.83M | 47.03 | ~1.1 GB | 4 (RGB-NIR) | 29.84 | 0.8412 | 0.0289 | 1.66° | 0.0245 |
| **TriSpecWaveMambaUQ (Ours)** | **6.17M** *(1.12M base)* | **24.96** | **270 MB** | **12 (All)** | **32.17** | **0.8754** | **0.0198** | **1.13°** | **0.0161** |
| *Performance vs. Bicubic* | — | — | — | — | *+2.13 dB* | *+0.0738* | *-18.2%* | *-18.2%* | *-28.7%* |
| *Performance vs. Teacher* | — | *1.88x Faster* | *4x Less VRAM* | *+8 Bands* | *+2.33 dB* | *+0.0342* | *-31.5%* | *-31.5%* | *-34.3%* |

As shown in Table II, TriSpecWaveMambaUQ outperforms Bicubic interpolation by **$+2.13\,\text{dB}$ PSNR** and **$+0.0738$ SSIM**, while cutting SAM spectral error by **$18.2\%$** and NDVI MAE by **$28.7\%$**. Compared to OpenSR-SRGAN, our architecture is nearly $2\times$ faster, uses $75\%$ less VRAM, and scales super-resolution to **all 12 bands simultaneously**.

### Table III: Per-Band Mean Absolute Error (MAE) across Native Sensor Groups

| Band Group | Band | Description | Wavelength ($\lambda$) | Native GSD | Bicubic MAE | Ours MAE | Improvement |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$10\,\text{m}$ VNIR** | B2 | Blue | $490\,\text{nm}$ | $10\,\text{m}$ | 0.0142 | **0.0094** | **-33.8%** |
| | B3 | Green | $560\,\text{nm}$ | $10\,\text{m}$ | 0.0158 | **0.0108** | **-31.6%** |
| | B4 | Red | $665\,\text{nm}$ | $10\,\text{m}$ | 0.0184 | **0.0121** | **-34.2%** |
| | B8 | Broad NIR | $842\,\text{nm}$ | $10\,\text{m}$ | 0.0245 | **0.0168** | **-31.4%** |
| **$20\,\text{m}$ Red Edge / SWIR** | B5 | Red Edge 1 | $705\,\text{nm}$ | $20\,\text{m}$ | 0.0210 | **0.0145** | **-31.0%** |
| | B6 | Red Edge 2 | $740\,\text{nm}$ | $20\,\text{m}$ | 0.0272 | **0.0191** | **-29.8%** |
| | B7 | Red Edge 3 | $783\,\text{nm}$ | $20\,\text{m}$ | 0.0298 | **0.0211** | **-29.2%** |
| | B8A | Narrow NIR | $865\,\text{nm}$ | $20\,\text{m}$ | 0.0312 | **0.0224** | **-28.2%** |
| | B11 | SWIR-1 | $1610\,\text{nm}$ | $20\,\text{m}$ | 0.0264 | **0.0182** | **-31.1%** |
| | B12 | SWIR-2 | $2190\,\text{nm}$ | $20\,\text{m}$ | 0.0221 | **0.0153** | **-30.8%** |
| **$60\,\text{m}$ Atmospheric** | B1 | Coastal Aerosol | $443\,\text{nm}$ | $60\,\text{m}$ | 0.0125 | **0.0089** | **-28.8%** |
| | B9 | Water Vapor | $945\,\text{nm}$ | $60\,\text{m}$ | 0.0195 | **0.0139** | **-28.7%** |

### Table IV: Biophysical Index Preservation Fidelity

| Index | Mathematical Formula | Physical Role | Bicubic MAE | Ours MAE | Improvement |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **NDVI** | $(\text{B8} - \text{B4}) / (\text{B8} + \text{B4})$ | Canopy greenness & vegetative vigour | 0.0226 | **0.0161** | **-28.7%** |
| **NDRE** | $(\text{B8} - \text{B5}) / (\text{B8} + \text{B5})$ | Chlorophyll content & nitrogen status | 0.0268 | **0.0189** | **-29.5%** |
| **NDWI** | $(\text{B3} - \text{B8}) / (\text{B3} + \text{B8})$ | Open surface water delineation | 0.0214 | **0.0148** | **-30.8%** |

### E. Qualitative Fidelity & Uncertainty Calibration
- **Structural Integrity:** Building footprints, road networks, and agricultural field boundaries exhibit crisp geometric separation without pixelation or high-frequency GAN texture noise.
- **Uncertainty Calibration:** Predicted aleatoric variance $\boldsymbol{\sigma}^2$ remains low in homogeneous terrain ($\sigma_b^2 < 0.002$ in water and uniform crop canopies) and cleanly spikes along complex parcel transitions, building edges ($\sigma_b^2 \in [0.015, 0.045]$), and semi-transparent cloud fringes. The Spearman rank correlation between $\boldsymbol{\sigma}^2$ and absolute residual error reaches $\rho = 0.764$, confirming well-calibrated confidence bounds.

### F. Architectural Ablation Studies

### Table V: Comprehensive Component Ablation Studies

| Configuration | Wavelet Decoupling | MTF Sensor Loss | Shifted-Window Loss | Uncertainty Head | PSNR (dB) ↑ | SSIM ↑ | SAM (rad) ↓ | $\text{MAE}_{\text{NDVI}}$ ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(A) Baseline CNN** | ✗ | ✗ | ✗ | ✗ | 30.62 | 0.8241 | 0.0235 | 0.0205 |
| **(B) + Wavelet + Mamba** | ✓ | ✗ | ✗ | ✗ | 31.45 | 0.8520 | 0.0218 | 0.0182 |
| **(C) + Shifted-Window Loss** | ✓ | ✗ | ✓ | ✗ | 31.78 | 0.8634 | 0.0210 | 0.0174 |
| **(D) + MTF Sensor Loss** | ✓ | ✓ | ✓ | ✗ | 32.04 | 0.8712 | 0.0201 | 0.0165 |
| **(E) Full Model (Ours)** | **✓** | **✓** | **✓** | **✓** | **32.17** | **0.8754** | **0.0198** | **0.0161** |

---

## V. Limitations & Future Work

### A. Architectural Limitations
1. **Stochastic Micro-Texture Smoothing:** Ultra-dense stochastic foliage (e.g., dense forest canopies) exhibits minor smoothing relative to unconstrained GAN outputs. However, this trade-off preserves radiometric integrity and is transparently flagged by the uncertainty head.
2. **Atmospheric Penetration:** As an optical method, dense opaque cloud cover attenuates surface radiance completely; the model accurately assigns maximum uncertainty to cloud plumes rather than synthesizing ungrounded terrain.

### B. Future Horizons
1. **Multimodal SAR-Optical Fusion:** Coupling Sentinel-1 C-band Synthetic Aperture Radar with Sentinel-2 MSI via cross-modal state-space attention to provide all-weather penetrating super-resolution.
2. **On-Orbit Space-Grade FPGA Deployment:** With a lightweight $1.12\,\text{M}$ base backbone and integer-friendly state space scans, the model is directly suited for on-satellite edge deployment (e.g., AMD-Xilinx Versal) prior to ground transmission.
3. **Diffusion-Mamba Hybridization:** Investigating Mamba-guided diffusion models to synthesize photo-realistic micro-textures while strictly adhering to sensor MTF bounds.

---

## VI. References

1. C. Dong, C. C. Loy, K. He, and X. Tang, "Image super-resolution using deep convolutional networks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 38, no. 2, pp. 295–307, 2015.
2. J. Kim, J. K. Lee, and K. M. Lee, "Accurate image super-resolution using very deep convolutional networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 1646–1654.
3. B. Lim, S. Son, H. Kim, S. Nah, and K. M. Lee, "Enhanced deep residual networks for single image super-resolution," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2017, pp. 136–144.
4. C. Ledig et al., "Photo-realistic single image super-resolution using a generative adversarial network," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 4681–4690.
5. C. Lanaras, J. Bioucas-Dias, S. Galliani, E. Baltsavias, and K. Schindler, "Super-resolution of Sentinel-2 images with deep residual networks," *ISPRS J. Photogramm. Remote Sens.*, vol. 146, pp. 305–316, 2018.
6. Y. Zhang, K. Li, K. Li, L. Wang, B. Zhong, and Y. Fu, "Image super-resolution using very deep residual channel attention networks," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 286–301.
7. X. Wang et al., "ESRGAN: Enhanced super-resolution generative adversarial networks," in *Proc. Eur. Conf. Comput. Vis. Workshops (ECCVW)*, 2018, pp. 63–79.
8. M. Deudon et al., "HighRes-net: Recursive fusion for multi-frame super-resolution of satellite imagery," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2020, pp. 188–189.
9. J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, "SwinIR: Image restoration using swin transformer," in *Proc. IEEE Int. Conf. Comput. Vis. Workshops (ICCVW)*, 2021, pp. 1833–1844.
10. S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, "Restormer: Efficient transformer for high-resolution image restoration," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 5728–5739.
11. L. Chen, X. Chu, X. Zhang, and J. Sun, "Simple baselines for image restoration," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2022, pp. 17–33.
12. S. Donike, C. Aybar, J. A. P. Gómez, D. Montero, and M. Camps-Valls, "OpenSR: A comprehensive benchmark for Sentinel-2 super-resolution," *IEEE Geosci. Remote Sens. Lett.*, vol. 20, pp. 1–5, 2023.
13. A. Kendall and Y. Gal, "What uncertainties do we need in Bayesian deep learning for computer vision?," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 5574–5584.
14. A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv preprint arXiv:2312.00752*, 2023.
15. L. Zhu et al., "Vision Mamba: Efficient visual representation learning with bidirectional state space model," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2024.
16. Y. Liu et al., "VMamba: Visual state space model," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2024.
17. H. Guo et al., "MambaIR: A simple baseline for image restoration with state space model," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2024.
18. J. W. Rouse, R. H. Haas, J. A. Schell, and D. W. Deering, "Monitoring vegetation systems in the Great Plains with ERTS," *NASA Spec. Publ.*, vol. 351, p. 309, 1974.
19. A. A. Gitelson, A. Viña, V. Ciganda, D. C. Rundquist, and T. J. Arkebauer, "Remote estimation of canopy chlorophyll content in crops," *Geophys. Res. Lett.*, vol. 32, no. 8, 2005.
20. B.-C. Gao, "NDWI—A normalized difference water index for remote sensing of vegetation liquid water from space," *Remote Sens. Environ.*, vol. 58, no. 3, pp. 257–266, 1996.
21. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE Trans. Image Process.*, vol. 13, no. 4, pp. 600–612, 2004.
22. European Space Agency (ESA), "Sentinel-2 MSI Level-2A Algorithm Theoretical Basis Document (ATBD)," *ESA SP-1322/2*, Tech. Rep., 2021.
