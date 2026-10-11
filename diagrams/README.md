# TriSpecWave-Mamba-UQ: Architecture & Pipeline Diagrams

This directory contains high-resolution diagrams of the **TriSpecWave-Mamba-UQ** architecture, including color-coded module attributions and horizontal single-page pipeline views.

---

## 1. Colored Architecture Pipeline (Recommended for Presentations & Reports)

* **PNG Image (300 DPI)**: [`TriSpecWave_Mamba_UQ_Colored_Pipeline.png`](./TriSpecWave_Mamba_UQ_Colored_Pipeline.png)
* **HTML Source**: [`TriSpecWave_Mamba_UQ_Colored_Pipeline.html`](./TriSpecWave_Mamba_UQ_Colored_Pipeline.html)

### Color-Coded Component Mapping:

| Module / Component | Visual Color Tag | Where It Appears in Pipeline | Role in Architecture |
| :--- | :--- | :--- | :--- |
| **2D DWT / Wavelet Transform** | `Sky Blue` | Column 2 (Card 1) | Separates low-frequency (`LL`) from high-frequency edges (`LH, HL, HH`). Mathematical transform. |
| **VSSD / Mamba-2 blocks** | `Amber / Gold` | Column 2 (LL Branch) | Models long-range 2D spatial relationships (fields, rivers, urban grid) with linear complexity. |
| **NAFBlock / StripConv** | `Emerald Green` | Column 2 (HF Branch) | Directional `1×K` & `K×1` depthwise convs for linear structures, boundaries, and roads. |
| **Spectral Encoder / Spectral Mamba** | `Amber / Gold` | Column 2 (Card 3) | Learns spectral dependencies across Sentinel-2 bands using physical wavelength ($\lambda$) and GSD tokens. |
| **Cross-Resolution Gated Fusion** | `Rose / Coral` | Column 3 (Card 1) | Combines 10m, 20m, and 60m branches, using sharp 10m boundaries to spatially allocate coarser bands. |
| **Spectral Graph Mixer** | `Purple` | Column 3 (Card 2) | Graph neural network treating bands as connected nodes across VIS, Red-Edge, NIR, and SWIR. |
| **Cross-Frequency Fusion + IDWT** | `Sky Blue` | Column 3 (Card 3) | Inverse Wavelet Synthesis recombining processed LL coarse and HF edge components back to spatial domain. |
| **PixelShuffle ×2 (Stage 1)** | `Violet` | Column 4 (Card 1) | First progressive sub-pixel upsampling step ($10\text{ m} \rightarrow 5\text{ m}$). |
| **NAFBlock / NAFNet Refinement** | `Emerald Green` | Column 4 (Card 2) | Mid-scale convolutional denoising, edge restoration, and detail recovery at 5 m GSD. |
| **PixelShuffle ×2 (Stage 2)** | `Violet` | Column 4 (Card 3) | Final sub-pixel upsampling step ($5\text{ m} \rightarrow 2.5\text{ m}$). |
| **Heteroscedastic Uncertainty Head** | `Orange` | Column 5 (Card 2) | Predicts mean $\mu$ and band-specific variance $\sigma^2$, generating 12 aleatoric uncertainty maps. |
| **MC-Dropout / Stochastic Mamba** | `Orange` | Column 5 (Card 3) | Estimates epistemic model uncertainty across stochastic passes to flag OOD and cloud artifacts. |
| **MTF / PSF Sensor Model** | `Ruby Red` | Column 6 (All Cards) | Physics forward degradation downsampling 2.5m predictions back to 10m, 20m, 60m to enforce sensor consistency. |

---

## 2. Additional Pipeline Formats

* **Pure Video Replica (Unbranded 6 Columns)**: [`Sentinel2_Video_Exact_Pure.png`](./Sentinel2_Video_Exact_Pure.png)
* **Video Layout with Header Banner**: [`Sentinel2_Video_Exact_Pipeline.png`](./Sentinel2_Video_Exact_Pipeline.png)
* **Single-Trunk Technical Hierarchy**: [`TriSpecWave_Mamba_UQ_Exact.png`](./TriSpecWave_Mamba_UQ_Exact.png)
* **High-Level Simplified Flowchart**: [`Simplified_Model_Overview.png`](./Simplified_Model_Overview.png)
* **Full Technical Deep-Dive Diagram**: [`TriSpecWave_Mamba_UQ_Architecture.png`](./TriSpecWave_Mamba_UQ_Architecture.png)
