import os
import sys
from playwright.sync_api import sync_playwright

diagram1_mmd = """flowchart TD
    classDef inputStyle fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef branch10 fill:#0284c7,stroke:#bae6fd,stroke-width:2px,color:#ffffff;
    classDef branch20 fill:#15803d,stroke:#bbf7d0,stroke-width:2px,color:#ffffff;
    classDef branch60 fill:#7e22ce,stroke:#f3e8ff,stroke-width:2px,color:#ffffff;
    classDef mambaStyle fill:#d97706,stroke:#fef3c7,stroke-width:2px,color:#ffffff;
    classDef fusionStyle fill:#b91c1c,stroke:#fecaca,stroke-width:2px,color:#ffffff;
    classDef decoderStyle fill:#4338ca,stroke:#e0e7ff,stroke-width:2px,color:#ffffff;
    classDef headStyle fill:#047857,stroke:#a7f3d0,stroke-width:2px,color:#ffffff;
    classDef outStyle fill:#1e293b,stroke:#22c55e,stroke-width:3px,color:#f8fafc;
    classDef uqStyle fill:#1e293b,stroke:#f59e0b,stroke-width:3px,color:#f8fafc;
    classDef lossStyle fill:#881337,stroke:#fda4af,stroke-width:2px,color:#ffffff;

    subgraph S2_INPUTS ["1. Sentinel-2 L2A Native-Resolution Inputs"]
        IN_10M["10m Native GSD (4 Bands)<br/>B2 (Blue 493nm), B3 (Green 560nm)<br/>B4 (Red 665nm), B8 (NIR 833nm)<br/>[B, 4, H, W]"]:::branch10
        IN_20M["20m Native GSD (6 Bands)<br/>B5 (704nm), B6 (740nm), B7 (783nm)<br/>B8A (865nm), B11 (1614nm), B12 (2202nm)<br/>[B, 6, H/2, W/2]"]:::branch20
        IN_60M["60m Native GSD (2 Bands)<br/>B1 (Coastal 443nm), B9 (Water Vapor 945nm)<br/>(B10 Cirrus excluded in L2A)<br/>[B, 2, H/6, W/6]"]:::branch60
    end

    subgraph BRANCH_10M ["2A. Spatial High-Resolution Encoder (10m Native)"]
        DWT["2D Discrete Wavelet Transform (DWT)<br/>Separates LL and HF subbands"]:::branch10
        LL_STREAM["LL Coarse Stream<br/>2D Visual State Space Dual (VSSD Mamba)<br/>Long-range contextual dependencies"]:::mambaStyle
        HF_STREAM["High-Frequency (LH, HL, HH) Stream<br/>Directional CNN / NAF-Blocks<br/>Edge and texture preservation"]:::branch10
        FUSE_10M["Wavelet Inverse / HR Spatial Fusion<br/>F_10m in R^[B, C1, H, W]"]:::branch10
    end

    subgraph BRANCH_20M ["2B. Red-Edge & SWIR Spectral Encoder (20m Native)"]
        WAVELENGTH_EMB["Physical Spectral Tokenizer<br/>Central Wavelength λ + Bandwidth Δλ + GSD<br/>Learned Continuous Wavelength Embedding"]:::branch20
        SPEC_MAMBA["Bidirectional Spectral Mamba Block<br/>Ordered Sequence: Blue ↔ Green ↔ Red ↔ RedEdge ↔ NIR ↔ SWIR<br/>Selective State Space over Spectral Channels"]:::mambaStyle
        FEAT_20M["RedEdge/SWIR Latent Representation<br/>F_20m in R^[B, C2, H/2, W/2]"]:::branch20
    end

    subgraph BRANCH_60M ["2C. Atmospheric Context Encoder (60m Native)"]
        ATM_ENC["Atmospheric Context Encoder<br/>Lightweight Mamba / MLP Layers<br/>Broad aerosol & vapor column prior"]:::branch60
        FEAT_60M["Atmospheric Latent Features<br/>F_60m in R^[B, C3, H/6, W/6]"]:::branch60
    end

    subgraph GRAPH_MODULE ["3. Cross-Band Spectral Graph Mixer"]
        GRAPH_CONSTRUCT["Dynamic Adjacency Matrix A_ij<br/>Weights = f( |Δλ|, Cosine Sim, Spatial Corr )"]:::mambaStyle
        GRAPH_GNN["Spectral Message Passing / GCN Layer<br/>Explicit physical graph pathways:<br/>• B2-B3-B4-B8 Visible/NIR cluster<br/>• B5-B6-B7-B8A Red-Edge slope cluster<br/>• B11-B12 SWIR absorption cluster<br/>• B1, B9 Atmospheric baseline"]:::mambaStyle
    end

    subgraph GUIDED_FUSION ["4. Progressive Cross-Resolution Guided Fusion"]
        FUSE_60_20["Guided Cross-Attention Fusion (60m → 20m)<br/>Scale Factor: ×3<br/>20m RedEdge/SWIR features guide 60m upsampling<br/>Prevents unconstrained 24× spatial hallucination"]:::fusionStyle
        FUSE_20_10["Spatial Allocation Guided Fusion (20m → 10m)<br/>Scale Factor: ×2<br/>10m B4 (Red) & B8 (NIR) high-contrast boundaries<br/>guide RedEdge & SWIR spatial allocation"]:::fusionStyle
        UNIFIED_10M["Unified 12-Band Joint Feature Tensor @ 10m<br/>F_joint in R^[B, C_fused, H, W]"]:::fusionStyle
    end

    subgraph SR_DECODER ["5. Progressive Wavelet-NAF Decoder"]
        UP_5M["PixelShuffle Upsampling (×2) → 5m GSD<br/>[B, C_d1, 2H, 2W]"]:::decoderStyle
        NAF_5M["NAF Refinement Block @ 5m<br/>Nonlinear Activation Free residual filtering"]:::decoderStyle
        UP_2_5M["PixelShuffle Upsampling (×2) → 2.5m GSD<br/>[B, C_d2, 4H, 4W]"]:::decoderStyle
        NAF_2_5M["Wavelet Synthesis & High-Order NAF Block @ 2.5m<br/>Sub-pixel structure restoration"]:::decoderStyle
        CROSS_BAND_CONSIST["Cross-Band Consistency & Index Regularizer Block<br/>Couples gradients across all 12 bands"]:::decoderStyle
    end

    subgraph SPEC_HEADS ["6. Domain-Specialized Reconstruction Heads (2.5m)"]
        VIS_HEAD["Visible Spectrum Head<br/>Target: B2, B3, B4 (RGB)<br/>Focus: Spatial sharpness, edge contrast"]:::headStyle
        RE_NIR_HEAD["Red-Edge & NIR Head<br/>Target: B5, B6, B7, B8, B8A<br/>Focus: Chlorophyll edge steepness, biomass curves"]:::headStyle
        SWIR_ATM_HEAD["SWIR & Atmospheric Head<br/>Target: B11, B12, B1, B9<br/>Focus: Moisture absorption, geology, aerosol removal"]:::headStyle
    end

    subgraph OUTPUT_SPACE ["7. Dual-Task Output Space (2.5m Target GSD)"]
        SR_OUT["12-Band Super-Resolved Reflectance (μ)<br/>μ_B1, μ_B2, μ_B3, μ_B4, μ_B5, μ_B6,<br/>μ_B7, μ_B8, μ_B8A, μ_B9, μ_B11, μ_B12<br/>Spatial Resolution: 2.5m GSD"]:::outStyle
        ALEATORIC_UQ["12-Band Aleatoric Uncertainty (σ²)<br/>Band-Specific Variance Maps:<br/>• B2,B3,B4,B8: LOW uncertainty (4× factor)<br/>• B5-B7,B8A,B11,B12: MEDIUM uncertainty (8× factor)<br/>• B1,B9: HIGH uncertainty (24× factor)"]:::uqStyle
        EPISTEMIC_UQ["Shared Epistemic Uncertainty<br/>Latent variance / MC-Dropout / Ensemble Head<br/>Identifies Out-Of-Distribution (OOD) / cloud artifacts"]:::uqStyle
    end

    subgraph PHYSICS_LOSS ["8. Physical Sensor PSF Consistency & Loss Objectives"]
        PSF_10M["Sensor PSF Downsampler D_10m(·)<br/>Convolves SR B2,B3,B4,B8 with 10m MSI MTF"]:::lossStyle
        PSF_20M["Sensor PSF Downsampler D_20m(·)<br/>Convolves SR B5-B7,B8A,B11,B12 with 20m MSI MTF"]:::lossStyle
        PSF_60M["Sensor PSF Downsampler D_60m(·)<br/>Convolves SR B1,B9 with 60m MSI MTF"]:::lossStyle

        TOTAL_LOSS["Total Optimization Objective L<br/>L = L_SR + λ1·L_SAM + λ2·L_wavelet + λ3·L_sensor + λ4·L_index + λ5·L_UQ"]:::lossStyle
    end

    IN_10M --> DWT
    DWT --> LL_STREAM
    DWT --> HF_STREAM
    LL_STREAM --> FUSE_10M
    HF_STREAM --> FUSE_10M

    IN_20M --> WAVELENGTH_EMB
    WAVELENGTH_EMB --> SPEC_MAMBA
    SPEC_MAMBA --> FEAT_20M

    IN_60M --> ATM_ENC
    ATM_ENC --> FEAT_60M

    FUSE_10M -.-> GRAPH_CONSTRUCT
    FEAT_20M -.-> GRAPH_CONSTRUCT
    FEAT_60M -.-> GRAPH_CONSTRUCT
    GRAPH_CONSTRUCT --> GRAPH_GNN

    FEAT_60M --> FUSE_60_20
    FEAT_20M --> FUSE_60_20
    GRAPH_GNN --> FUSE_60_20

    FUSE_60_20 -->|"20m enriched features"| FUSE_20_10
    FUSE_10M -->|"10m spatial boundary guidance"| FUSE_20_10

    FUSE_20_10 --> UNIFIED_10M

    UNIFIED_10M --> UP_5M
    UP_5M --> NAF_5M
    NAF_5M --> UP_2_5M
    UP_2_5M --> NAF_2_5M
    NAF_2_5M --> CROSS_BAND_CONSIST

    CROSS_BAND_CONSIST --> VIS_HEAD
    CROSS_BAND_CONSIST --> RE_NIR_HEAD
    CROSS_BAND_CONSIST --> SWIR_ATM_HEAD

    VIS_HEAD -->|"μ_B2, μ_B3, μ_B4"| SR_OUT
    RE_NIR_HEAD -->|"μ_B5, μ_B6, μ_B7, μ_B8, μ_B8A"| SR_OUT
    SWIR_ATM_HEAD -->|"μ_B1, μ_B9, μ_B11, μ_B12"| SR_OUT

    CROSS_BAND_CONSIST -->|"Heteroscedastic Head"| ALEATORIC_UQ
    CROSS_BAND_CONSIST -->|"Latent Variance Head"| EPISTEMIC_UQ

    SR_OUT -->|"B2, B3, B4, B8"| PSF_10M
    SR_OUT -->|"B5, B6, B7, B8A, B11, B12"| PSF_20M
    SR_OUT -->|"B1, B9"| PSF_60M

    PSF_10M -.->|"Compare with Native 10m Input"| TOTAL_LOSS
    PSF_20M -.->|"Compare with Native 20m Input"| TOTAL_LOSS
    PSF_60M -.->|"Compare with Native 60m Input"| TOTAL_LOSS
    ALEATORIC_UQ -.->|"Heteroscedastic NLL"| TOTAL_LOSS
    SR_OUT -.->|"SAM + Wavelet + Spectral Indices"| TOTAL_LOSS
"""

diagram2_mmd = """flowchart LR
    classDef stageStyle fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#ffffff;
    classDef dataStyle fill:#064e3b,stroke:#34d399,stroke-width:1px,color:#ffffff;

    subgraph CURRICULUM ["6-Stage Progressive Training Strategy"]
        S0["Stage 0: Pretraining<br/>Representation Learning"]:::stageStyle
        S1["Stage 1: Spatial SR<br/>10m B2,B3,B4,B8 → 2.5m"]:::stageStyle
        S2["Stage 2: Wald Self-Supervision<br/>20m B5-B7,B8A,B11,B12 → 10m"]:::stageStyle
        S3["Stage 3: Joint 12-Band SR<br/>Full 2.5m Fine-Tuning"]:::stageStyle
        S4["Stage 4: Physics & Sensor Consistency<br/>PSF + Wavelet + Index Loss"]:::stageStyle
        S5["Stage 5: Calibration<br/>UQ & Out-of-Distribution"]:::stageStyle
    end

    subgraph DATA_SOURCES ["Curated Datasets Stack"]
        D0["SSL4EO-S12 v1.1 + EnMAP/SpectralEarth"]:::dataStyle
        D1["SEN2NAIPv2 + SEN2VENµS + WorldStrat"]:::dataStyle
        D2["CloudSEN12 / CloudSEN12+ (Cloud-free subset)"]:::dataStyle
        D3["SEN2NEON (2,269 tiles) + Custom AVIRIS-NG ↔ S2 Pairs"]:::dataStyle
        D4["Copernicus S2 L2A Global Stratified Corpus"]:::dataStyle
        D5["OpenSR-Test + MuS2 + Diverse Biomes"]:::dataStyle
    end

    D0 --> S0
    S0 --> S1
    D1 --> S1
    S1 --> S2
    D2 --> S2
    S2 --> S3
    D3 --> S3
    S3 --> S4
    D4 --> S4
    S4 --> S5
    D5 --> S5
"""

os.makedirs("diagrams", exist_ok=True)

with open(r"diagrams\TriSpecWave_Mamba_UQ_Architecture.mmd", "w", encoding="utf-8") as f:
    f.write(diagram1_mmd.strip())

with open(r"diagrams\TriSpecWave_Mamba_UQ_Training_Curriculum.mmd", "w", encoding="utf-8") as f:
    f.write(diagram2_mmd.strip())

html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="../tools/mermaid.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            padding: 30px;
            background-color: #0b0f19;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            display: inline-block;
            min-width: 100%;
        }
        #chart-wrapper {
            display: inline-block;
            background-color: #0f172a;
            padding: 30px;
            border-radius: 16px;
            border: 1px solid #1e293b;
        }
    </style>
</head>
<body>
    <div id="chart-wrapper">
        <pre class="mermaid">
DIAGRAM_CODE_HERE
        </pre>
    </div>
    <script>
        mermaid.initialize({
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            flowchart: {
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis'
            }
        });
    </script>
</body>
</html>
"""

def render_diagram(name, mmd_code):
    html_content = html_template.replace("DIAGRAM_CODE_HERE", mmd_code.strip())
    html_path = os.path.abspath(f"diagrams/{name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    png_path = os.path.abspath(f"diagrams/{name}.png")
    svg_path = os.path.abspath(f"diagrams/{name}.svg")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            headless=True
        )
        # Start with generous viewport so initial layout calculates properly
        context = browser.new_context(viewport={"width": 3840, "height": 3840}, device_scale_factor=2)
        page = context.new_page()
        page.goto(f"file:///{html_path}")
        
        # Wait for SVG to render
        page.wait_for_selector(".mermaid svg", timeout=15000)
        
        # Extract SVG outer HTML
        svg_element = page.locator(".mermaid svg")
        svg_content = svg_element.evaluate("el => el.outerHTML")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
            
        # Get true bounding box of the chart wrapper
        wrapper = page.locator("#chart-wrapper")
        box = wrapper.bounding_box()
        
        # Adjust viewport to fit entire wrapper exactly with margin
        needed_w = int(box["x"] + box["width"] + 60)
        needed_h = int(box["y"] + box["height"] + 60)
        page.set_viewport_size({"width": max(needed_w, 1920), "height": max(needed_h, 1080)})
        
        # Screenshot the wrapper element directly
        wrapper.screenshot(path=png_path)
        browser.close()

    print(f"Rendered {name}:")
    print(f"  PNG: {png_path} ({os.path.getsize(png_path)} bytes)")
    print(f"  SVG: {svg_path} ({os.path.getsize(svg_path)} bytes)")

render_diagram("TriSpecWave_Mamba_UQ_Architecture", diagram1_mmd)
render_diagram("TriSpecWave_Mamba_UQ_Training_Curriculum", diagram2_mmd)
print("Rendering completed cleanly without any clipping!")
