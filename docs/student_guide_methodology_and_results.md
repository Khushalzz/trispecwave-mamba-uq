# A 3rd-Year CSE Student's Guide to TriSpecWaveMambaUQ
## Understanding State Space Super-Resolution, Wavelet Decoupling, and Calibrated Uncertainty in Satellite Computer Vision

---

### Welcome to the Lecture!
Imagine you are building a computer vision system for satellites orbiting 786 kilometers above the Earth. Your camera captures images of our planet, but physics gets in the way: optical sensors must trade off between light-gathering ability, sensor size, and swath width. 

As a result, the European Space Agency's **Sentinel-2** satellite doesn't give you clean, high-resolution RGB pictures. Instead, it captures **12 distinct spectral bands of light** distributed across three completely different spatial grid resolutions:
- **10-meter resolution (Sharp):** Blue, Green, Red, and Near-Infrared (B2, B3, B4, B8).
- **20-meter resolution (Medium):** Red Edge and Short-Wave Infrared (B5, B6, B7, B8A, B11, B12).
- **60-meter resolution (Coarse):** Atmospheric bands (B1 Coastal Aerosol, B9 Water Vapor).

**Our Goal as Computer Scientists:**  
Take all 12 heterogeneous bands and build a deep learning model that super-resolves them by a factor of $4\times$—downscaling the spatial grid from **$10\,\text{m}$ to $2.5\,\text{m}$ Ground Sampling Distance (GSD)**—while running in real-time on standard GPUs, maintaining 100% radiometric physics fidelity, and outputting a confidence score for every single pixel.

This guide breaks down the architecture and results of **TriSpecWaveMambaUQ** using core Computer Science concepts you already know: Big-O notation, dynamic programming, recurrence relations, frequency domain processing, and probabilistic loss functions.

---

## 1. The Core Engineering Dilemmas in Satellite Vision

Before writing any neural network layer, let's understand why standard deep learning architectures fail at this task:

```
+-----------------------------------------------------------------------------------------+
|                               FOUR CORE CS BOTTLENECKS                                  |
+-----------------------------------------------------------------------------------------+
| 1. Input Mismatch:   Standard CNNs expect 3-channel RGB of identical size. Sentinel-2  |
|                      has 12 channels at 10m, 20m, and 60m resolutions.                  |
| 2. Receptive Field:  Standard CNNs have tiny local receptive fields (3x3 kernels).      |
|                      Vision Transformers have global receptive fields, but suffer from  |
|                      O(N²) quadratic attention complexity—blowing up GPU memory (OOM).  |
| 3. Hallucination:    Standard Super-Resolution GANs (like ESRGAN) hallucinate fake     |
|                      textures (inventing non-existent roads or trees), corrupting       |
|                      scientific climate measurements.                                   |
| 4. Reliability Void: Standard neural networks are deterministic black boxes; they make   |
|                      blind guesses with 0% awareness of their own uncertainty.          |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Methodology: How TriSpecWaveMambaUQ Works

The system architecture solves these four bottlenecks through five modular pipeline stages:

```
========================================================================================
                          TriSpecWaveMambaUQ ARCHITECTURE PIPELINE
========================================================================================
 Input: 12 Sentinel-2 Bands [4 @ 10m  |  6 @ 20m  |  2 @ 60m]
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
     [10m Group Conv]       [20m Group Conv]       [60m Group Conv]
            │                      │                      │
            │                 Bilinear 2x            Bilinear 6x
            │                      │                      │
            └──────────────────────┴──────────────────────┘
                                   │
                       [Gated Bilinear Fusion]
                                   │
                   [Spectral Graph Attention] (3-Node)
                                   │
                [SpectralMamba 1D] (Continuous Wavelengths)
                                   │
                  [2D Haar Wavelet Decomposition]
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
     Low-Frequency (LL Blur)               High-Frequency (LH, HL, HH Edges)
     [2D SpatialMamba]                     [Directional StripConvs 1x7, 7x1]
     Global Context: O(N) Linear           Traces Linear Parcel & Road Edges
                └──────────────────┬──────────────────┘
                                   │
                       [Inverse Haar Wavelet (IDWT)]
                                   │
                   [Progressive PixelShuffle Upsampler]
                                (2x -> 4x)
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
     [Mean Head: μ]                        [Uncertainty Head: log σ²]
     12 Bands @ 2.5m GSD                   Pixel-Wise Confidence Map
                                   │
                       [Physics-Informed Losses]
        (MTF Optical Blur Check + Heteroscedastic NLL + SAM Color Check)
========================================================================================
```

---

### Stage A: Tri-Stream Grouped Encoder & Gated Fusion
Instead of naively resizing coarse 20m and 60m bands with bicubic interpolation (which destroys subtle spectral information), we organize the input into three streams matching physical sensor hardware:
- Group 1 ($10\,\text{m}$): 4 bands $\rightarrow$ Group Conv
- Group 2 ($20\,\text{m}$): 6 bands $\rightarrow$ Group Conv
- Group 3 ($60\,\text{m}$): 2 bands $\rightarrow$ Group Conv

Once initial features are extracted at native scales, the 20m and 60m features are upsampled to the 10m grid and fused through a **learnable gating mechanism**:
$$\mathbf{F}_{\text{fused}} = \mathbf{F}_{10\text{m}} + \mathbf{W}_{\text{gate}} \ast [\mathbf{F}_{20\text{m}}^{\uparrow}, \mathbf{F}_{60\text{m}}^{\uparrow}]$$

**Intuition:** The neural network learns a dynamic "attention gate" that decides exactly how much coarse atmospheric context to blend into the sharp 10m spatial features.

---

### Stage B: Cross-Spectral Modeling with SpectralMamba
The 12 channels are not arbitrary features; they are physical wavelengths of light along the electromagnetic spectrum ($443\,\text{nm}$ to $2190\,\text{nm}$). To capture how light reflects across different wavelengths, we need cross-channel modeling.

#### What is Mamba? (An Intuitive CS Primer)
In your Deep Learning classes, you learned about two paradigms:
1. **Recurrent Neural Networks (RNNs):** Fast linear inference $\mathcal{O}(N)$ memory, but slow sequential training and prone to vanishing gradients because transition weights are fixed.
2. **Transformers (Self-Attention):** Superb global memory and parallel training, but quadratic complexity $\mathcal{O}(N^2)$ in time and memory. If your image has $N = 256 \times 256 = 65,536$ pixels, an attention matrix contains $4.29 \times 10^9$ elements!

**Mamba (Selective State Space Model)** gives us the best of both worlds:
- It behaves like a continuous recurrence:
  $$h_t = \mathbf{A}_t h_{t-1} + \mathbf{B}_t x_t, \quad y_t = \mathbf{C}_t h_t$$
- **Key Innovation:** Unlike a classic RNN where matrices $\mathbf{A}, \mathbf{B}$ are static weights, Mamba makes $\mathbf{A}_t, \mathbf{B}_t, \mathbf{C}_t$ **input-dependent functions** of the current token $x_t$. The model can dynamically choose to remember important information or flush irrelevant data.
- **Hardware Efficiency:** Using parallel prefix sums (associative scan algorithms from parallel computing), Mamba can be computed in parallel during training while executing in **strictly linear time $\mathcal{O}(N)$** with a constant memory footprint during inference.

In **SpectralMamba**, we sort the channels by wavelength and run a 1D selective scan across all 12 bands. This captures full-spectrum radiative correlations in $\mathcal{O}(C)$ complexity instead of $\mathcal{O}(C^2)$ transformer overhead.

---

### Stage C: Frequency-Decoupled Spatial Processing (Haar Wavelets)
Real satellite images contain two completely distinct types of visual information:
1. **Low frequencies:** Broad, smooth illumination across plains, lakes, and large forests.
2. **High frequencies:** Sharp, razor-thin structural edges like roads, irrigation canals, and building outlines.

Trying to process both with the same convolutional filter is suboptimal. Instead, we use a **2D Haar Discrete Wavelet Transform (HaarDWT)**:

```
[Input Feature Map] ──(Haar DWT)──>  ┌───────┬───────┐
                                     │  LL   │  LH   │  <-- LL: Low-freq smooth blur
                                     ├───────┼───────┤  <-- LH, HL, HH: High-freq edges
                                     │  HL   │  HH   │      (Horizontal, Vertical, Diagonal)
                                     └───────┴───────┘
```

**Why this is brilliant engineering:**
1. **Low-Frequency Branch ($LL$):** Routed to **2D SpatialMamba**. Since the smooth background requires understanding global geographic context (e.g., distinguishing an entire farm from a reservoir), SpatialMamba scans across both spatial dimensions in **linear time $\mathcal{O}(N)$**, capturing the entire field of view without memory exhaustion.
2. **High-Frequency Branches ($LH, HL, HH$):** Routed to **Directional Strip Convolutions** ($1\times7$ and $7\times1$ kernels) and NAF (Nonlinear Activation Free) blocks. Roads and field borders run horizontally and vertically. Asymmetric strip filters act like specialized digital edge-tracers, sharpening linear infrastructure without adding unnecessary parameter bloat.
3. **Recombination:** An **Inverse Haar Wavelet Transform (IDWT)** losslessly combines the processed frequencies back into a single unified feature map.

---

### Stage D: Dual Output Heads (Mean and Uncertainty)
The unified features pass through a progressive $4\times$ PixelShuffle upsampler ($64 \times 64 \rightarrow 128 \times 128 \rightarrow 256 \times 256$) into two specialized heads:
1. **Mean Head ($\boldsymbol{\mu}$):** Predicts the reconstructed surface reflectance for all 12 bands at $2.5\,\text{m}$ resolution.
2. **Uncertainty Head ($\log \boldsymbol{\sigma}^2$):** Predicts a per-pixel confidence map (aleatoric variance).

**The Analogy:** Think of a student taking an engineering exam. For every question, the student gives both an answer ($\boldsymbol{\mu}$) and an honest confidence score ($\boldsymbol{\sigma}^2$). If a region has heavy shadow or ambiguous cloud fringes, the network outputs: *"Here is my best guess, but my uncertainty is high."*

---

### Stage E: The Loss Functions (Physics-Informed Sanity Checks)

How do we train this network without falling into the "GAN hallucination" trap? We use three intuitive loss terms:

#### 1. Heteroscedastic Negative Log-Likelihood (NLL) Loss
$$\mathcal{L}_{\text{hetero}} = \frac{1}{2} \sum_{\text{pixels}} \left( \frac{\vert y_{\text{true}} - \mu \vert}{\sigma^2} + \log \sigma^2 \right)$$
- If the model is confident ($\sigma^2$ is small), any prediction error is penalized heavily.
- If the pixel is inherently ambiguous (e.g., a fuzzy cloud fringe), the model can increase $\sigma^2$ to soften the penalty.
- The $\log \sigma^2$ term acts as a penalty regularizer: it prevents the model from cheating by simply predicting infinite uncertainty everywhere!

#### 2. Sensor MTF (Modulation Transfer Function) Degradation Loss
**The Physics Sanity Check:** If our $2.5\,\text{m}$ super-resolved image is physically correct, then simulating the satellite camera's optical blur (Point Spread Function) and downsampling it back to $10\,\text{m}$ **must recreate the original low-resolution satellite image**:
$$\mathcal{L}_{\text{MTF}} = \left\Vert \text{Downsample}\Big(\text{OpticalBlur}(\boldsymbol{\mu})\Big) - \mathbf{X}_{\text{low-res}} \right\Vert_1$$
This is a closed-loop consistency constraint. If the model tries to hallucinate a fake building or road that was never in the original satellite capture, this loss immediately penalizes it!

#### 3. Spectral Angle Mapper (SAM) Loss
Treats the 12 predicted bands at each pixel as a 12-dimensional vector and measures the angular difference against the ground truth:
$$\mathcal{L}_{\text{SAM}} = \arccos\left( \frac{\boldsymbol{\mu} \cdot \mathbf{y}}{\Vert \boldsymbol{\mu} \Vert \, \Vert \mathbf{y} \Vert} \right)$$
This ensures the model preserves the exact spectroscopic signatures of vegetation, water, and soil without color distortion.

---

## 3. Experimental Setup & Benchmark Results

We evaluated the model on two rigorous benchmarks:
1. **Pune Metropolitan Region (Maharashtra, India):** Complex real-world terrain spanning urban settlements, fragmented agricultural plots, the Khadakwasla reservoir, and river corridors. Sliced into 1,400 training, 100 validation, and 50 held-out unseen test patches.
2. **ESA OpenSR SEN2NEON Benchmark:** Standardized international benchmark pairing Sentinel-2 observations with airborne hyperspectral ground truth.

---

### Quantitative Evaluation Metrics (A Quick Refresher)
- **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction error on a logarithmic decibel scale. Higher is better (+1 dB is noticeable; +2 dB is a major leap).
- **SSIM (Structural Similarity Index):** Measures perceptual structural similarity (edges, textures) from 0 to 1. Higher is better.
- **SAM (Spectral Angle Mapper):** Measures spectral color vector distortion in radians/degrees. Lower is better.
- **NDVI MAE:** Mean Absolute Error for the Normalized Difference Vegetation Index $(\text{NIR} - \text{Red}) / (\text{NIR} + \text{Red})$, the most critical metric for agricultural health. Lower is better.

---

### Benchmark Comparison Table

| Model Architecture | Parameters | Latency (ms) | Peak VRAM | Bands Processed | PSNR (dB) ↑ | SSIM ↑ | SAM (rad) ↓ | SAM (deg) ↓ | $\text{MAE}_{\text{NDVI}}$ ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Interpolation** | — | — | — | 12 | 30.04 | 0.8016 | 0.0242 | 1.39° | 0.0226 |
| **OpenSR-SRGAN (Teacher)**| 1.83M | 47.03 ms | ~1,100 MB | 4 (RGB-NIR only) | 29.84 | 0.8412 | 0.0289 | 1.66° | 0.0245 |
| **TriSpecWaveMambaUQ (Ours)**| **6.17M** *(1.117M base)* | **24.96 ms** | **270 MB** | **12 (All Bands)** | **32.17** | **0.8754** | **0.0198** | **1.13°** | **0.0161** |
| *Improvement vs. Bicubic* | — | — | — | — | *+2.13 dB* | *+0.0738* | *-18.2%* | *-18.2%* | *-28.7%* |
| *Improvement vs. Teacher* | — | *1.88x Faster*| *4x Less VRAM* | *+8 Extra Bands* | *+2.33 dB* | *+0.0342* | *-31.5%* | *-31.5%* | *-34.3%* |

---

### Why These Results Matter to Computer Scientists:
1. **$32.17\,\text{dB}$ PSNR (+2.13 dB gain):** Significantly outperforms simple interpolation and generative teachers while eliminating noisy artifacts.
2. **$0.0198\,\text{rad}$ SAM (18.2% reduction):** Demonstrates that the model preserves true physical surface reflectance, not just pleasing RGB colors.
3. **Vegetation Index Accuracy ($0.0161$ NDVI MAE):** A 28.7% reduction in error ensures agricultural scientists can accurately detect crop stress and chlorophyll levels at sub-field scales.
4. **$24.96\,\text{ms}$ Latency & $270\,\text{MB}$ VRAM:** Executes in real-time ($>40\,\text{FPS}$) on modern GPUs and can easily be deployed on resource-constrained edge devices or ground stations.

---

### Per-Band Accuracy Across Sensor Groups

| Band Group | Band ID | Name | Wavelength | Native GSD | Bicubic MAE | Ours MAE | Error Reduction |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$10\,\text{m}$ Visible & NIR** | B2 | Blue | $490\,\text{nm}$ | $10\,\text{m}$ | 0.0142 | **0.0094** | **-33.8%** |
| | B3 | Green | $560\,\text{nm}$ | $10\,\text{m}$ | 0.0158 | **0.0108** | **-31.6%** |
| | B4 | Red | $665\,\text{nm}$ | $10\,\text{m}$ | 0.0184 | **0.0121** | **-34.2%** |
| | B8 | Broad NIR | $842\,\text{nm}$ | $10\,\text{m}$ | 0.0245 | **0.0168** | **-31.4%** |
| **$20\,\text{m}$ Red Edge & SWIR** | B5 | Red Edge 1 | $705\,\text{nm}$ | $20\,\text{m}$ | 0.0210 | **0.0145** | **-31.0%** |
| | B6 | Red Edge 2 | $740\,\text{nm}$ | $20\,\text{m}$ | 0.0272 | **0.0191** | **-29.8%** |
| | B7 | Red Edge 3 | $783\,\text{nm}$ | $20\,\text{m}$ | 0.0298 | **0.0211** | **-29.2%** |
| | B8A | Narrow NIR | $865\,\text{nm}$ | $20\,\text{m}$ | 0.0312 | **0.0224** | **-28.2%** |
| | B11 | SWIR-1 | $1610\,\text{nm}$ | $20\,\text{m}$ | 0.0264 | **0.0182** | **-31.1%** |
| | B12 | SWIR-2 | $2190\,\text{nm}$ | $20\,\text{m}$ | 0.0221 | **0.0153** | **-30.8%** |
| **$60\,\text{m}$ Atmospheric** | B1 | Coastal Aerosol | $443\,\text{nm}$ | $60\,\text{m}$ | 0.0125 | **0.0089** | **-28.8%** |
| | B9 | Water Vapor | $945\,\text{nm}$ | $60\,\text{m}$ | 0.0195 | **0.0139** | **-28.7%** |

Every single band sees an error reduction between **28% and 34%**, proving that the tri-stream grouped encoder benefits both high-resolution and low-resolution sensor inputs equally.

---

### Ablation Study: What Each Module Contributes

To verify that every architectural design decision was justified, we performed systematic ablation experiments:

| Configuration | Wavelet + Mamba | Shifted-Window Loss | Sensor MTF Loss | Uncertainty Head | PSNR (dB) ↑ | SSIM ↑ | SAM (rad) ↓ | $\text{MAE}_{\text{NDVI}}$ ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(A) Standard Baseline CNN** | ✗ | ✗ | ✗ | ✗ | 30.62 | 0.8241 | 0.0235 | 0.0205 |
| **(B) + Wavelet + Mamba Backbone** | ✓ | ✗ | ✗ | ✗ | 31.45 | 0.8520 | 0.0218 | 0.0182 |
| **(C) + Shifted-Window Loss** | ✓ | ✓ | ✗ | ✗ | 31.78 | 0.8634 | 0.0210 | 0.0174 |
| **(D) + Sensor MTF Degradation Loss**| ✓ | ✓ | ✓ | ✗ | 32.04 | 0.8712 | 0.0201 | 0.0165 |
| **(E) Full TriSpecWaveMambaUQ** | **✓** | **✓** | **✓** | **✓** | **32.17** | **0.8754** | **0.0198** | **0.0161** |

**Observations:**
- Replacing standard convolutions with **Wavelet Decoupling + Mamba** yielded the single largest performance jump (**$+0.83\,\text{dB}$ PSNR**).
- Adding the **Sensor MTF Loss** provided an immediate $+0.26\,\text{dB}$ boost and dropped spectral distortion below $0.020\,\text{rad}$.
- The **Uncertainty Head** not only provides calibrated confidence maps, but also regularizes training, pushing final fidelity to **$32.17\,\text{dB}$ PSNR** and **$0.8754$ SSIM**.

---

### How Uncertainty Quantification (UQ) Behaves in Real Life
When examining the predicted uncertainty maps ($\boldsymbol{\sigma}^2$):
- **Smooth, Homogeneous Regions (e.g., lakes, flat fields):** Very low uncertainty ($\sigma^2 < 0.002$).
- **High-Frequency Geographic Boundaries (e.g., building contours, roads):** Moderate uncertainty ($\sigma^2 \in [0.015, 0.045]$), reflecting boundary ambiguity.
- **Semi-transparent Clouds & Haze:** Massive spikes in uncertainty ($\sigma^2 > 0.10$). The model automatically flags obscured regions, acting as a built-in cloud and shadow detector without requiring an extra classification network!
- **Calibration Check:** The Spearman rank correlation between predicted uncertainty $\boldsymbol{\sigma}^2$ and true absolute error is $\boldsymbol{\rho = 0.764}$, verifying strong empirical calibration.

---

## 4. Summary Cheat Sheet for CSE Students

1. **The Problem:** 12 Sentinel-2 satellite bands exist at three different resolutions (10m, 20m, 60m). We super-resolve all 12 bands to a uniform 2.5m GSD.
2. **Why Not Transformers?** Self-attention requires $\mathcal{O}(N^2)$ memory. Large satellite tiles cause out-of-memory errors.
3. **Why Mamba?** Mamba uses input-dependent state-space recurrences computed via parallel scans, giving global context in **linear time $\mathcal{O}(N)$**.
4. **Why Wavelets?** Haar wavelets cleanly separate low-frequency blur (processed by SpatialMamba for global context) from high-frequency edges (processed by directional strip convolutions for roads and boundaries).
5. **Why MTF Loss?** A closed-loop physics check: blurring and downsampling our super-resolved prediction must reproduce the original low-res satellite image, preventing generative hallucinations.
6. **Why UQ?** The model outputs both a mean reflectance value $\boldsymbol{\mu}$ and a variance $\boldsymbol{\sigma}^2$, providing a calibrated confidence score for every pixel.
7. **The Benchmark Numbers:**
   - **$32.17\,\text{dB}$ PSNR** (+2.13 dB over Bicubic)
   - **$0.8754$ SSIM**
   - **$0.0198\,\text{rad}$ SAM** ($1.13^\circ$, -18.2% spectral error)
   - **$24.96\,\text{ms}$ Latency** (Real-time at >40 FPS)
   - **$270\,\text{MB}$ Peak VRAM** (Fits on edge hardware)
   - **$6.17\,\text{M}$ Parameters** ($1.117\,\text{M}$ base backbone)
