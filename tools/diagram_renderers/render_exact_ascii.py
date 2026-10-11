import os
from playwright.sync_api import sync_playwright

exact_mmd = """flowchart TD
    classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#ffffff;
    classDef encStyle fill:#0f766e,stroke:#2dd4bf,stroke-width:2px,color:#ffffff;
    classDef dwtStyle fill:#0284c7,stroke:#93c5fd,stroke-width:2px,color:#ffffff;
    classDef trunkStyle fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#ffffff;
    classDef outStyle fill:#065f46,stroke:#34d399,stroke-width:2px,color:#ffffff;
    classDef uqStyle fill:#78350f,stroke:#fcd34d,stroke-width:2px,color:#ffffff;
    classDef checkStyle fill:#881337,stroke:#f43f5e,stroke-width:2px,color:#ffffff;

    subgraph S2_INPUT ["SENTINEL-2 MULTI-RESOLUTION INPUT"]
        direction LR
        IN_10M["<b>10 m</b><br/>B2 B3 B4 B8"]:::inputStyle
        IN_20M["<b>20 m</b><br/>B5 B6 B7 B8A<br/>B11 B12"]:::inputStyle
        IN_60M["<b>60 m</b><br/>B1 B9"]:::inputStyle
    end

    subgraph S2_ENCODERS ["TRIPLE ENCODER STAGE"]
        direction LR
        subgraph BRANCH_10M ["10m Branch"]
            direction TB
            ENC_10M["<b>Spatial-Frequency Encoder</b>"]:::encStyle
            DWT["DWT (2D-Wavelet)"]:::dwtStyle
            LL["<b>LL</b><br/>VSSD ×4"]:::dwtStyle
            HF["<b>LH / HL / HH</b><br/>NAF + StripConv"]:::dwtStyle
            DWT_MERGE["Merge Subbands"]:::dwtStyle
        end

        subgraph BRANCH_20M ["20m Branch"]
            direction TB
            ENC_20M["<b>Spectral Encoder</b><br/>+ λ / GSD Embed<br/>(Continuous Physics Tokenizer)"]:::encStyle
        end

        subgraph BRANCH_60M ["60m Branch"]
            direction TB
            ENC_60M["<b>Context Encoder</b><br/>(Lightweight Atmosphere Prior)"]:::encStyle
        end
    end

    FUSION["<b>Cross-Resolution Gated Fusion</b>"]:::trunkStyle
    GRAPH["<b>Spectral Graph Mixer</b>"]:::trunkStyle
    XFREQ["<b>Cross-Frequency Fusion</b>"]:::trunkStyle
    IDWT["<b>IDWT (Inverse Wavelet Synthesis)</b>"]:::trunkStyle
    SKIP["<b>Residual / Skip Fusion</b>"]:::trunkStyle
    PROG_REC["<b>Progressive Reconstruction</b><br/>×2 → 5 m → NAF Refinement → ×2 → 2.5 m"]:::trunkStyle

    subgraph DUAL_OUT ["2.5 m OUTPUT HEADS"]
        direction LR
        OUT_MU["<b>12-band μ</b><br/>Super-Resolved Reflectance"]:::outStyle
        OUT_SIGMA["<b>12-band σ²</b><br/>Aleatoric Uncertainty"]:::uqStyle
        OUT_EPI["<b>Epistemic UQ</b><br/>Model Confidence"]:::uqStyle
    end

    CHECK["<b>Band-Specific MTF / PSF Check</b><br/>10m / 20m / 60m Sensor Reconstruction Consistency"]:::checkStyle

    %% Data flow
    IN_10M --> ENC_10M
    IN_20M --> ENC_20M
    IN_60M --> ENC_60M

    ENC_10M --> DWT
    DWT --> LL
    DWT --> HF
    LL --> DWT_MERGE
    HF --> DWT_MERGE

    DWT_MERGE --> FUSION
    ENC_20M --> FUSION
    ENC_60M --> FUSION

    FUSION --> GRAPH
    GRAPH --> XFREQ
    XFREQ --> IDWT
    IDWT --> SKIP
    ENC_10M -.->|"High-Res Spatial Skip"| SKIP
    SKIP --> PROG_REC

    PROG_REC --> OUT_MU
    PROG_REC --> OUT_SIGMA
    PROG_REC --> OUT_EPI

    OUT_MU -.->|"Sensor PSF Loop"| CHECK
    OUT_SIGMA -.->|"Calibrate"| CHECK
"""

os.makedirs("diagrams", exist_ok=True)
with open(r"diagrams\TriSpecWave_Mamba_UQ_Exact.mmd", "w", encoding="utf-8") as f:
    f.write(exact_mmd.strip())

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="../tools/mermaid.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 30px;
            background-color: #0b0f19;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            display: inline-block;
            min-width: 100%;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .header h1 {{
            color: #38bdf8;
            font-size: 24px;
            margin: 0 0 6px 0;
            letter-spacing: 0.5px;
        }}
        .header p {{
            color: #94a3b8;
            font-size: 14px;
            margin: 0;
        }}
        #chart-wrapper {{
            display: inline-block;
            background-color: #0f172a;
            padding: 30px 40px;
            border-radius: 16px;
            border: 1px solid #1e293b;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        }}
    </style>
</head>
<body>
    <div id="chart-wrapper">
        <div class="header">
            <h1>TriSpecWave-Mamba-UQ Architecture</h1>
            <p>Single-Trunk Progressive Multi-Resolution Sentinel-2 Super-Resolution</p>
        </div>
        <pre class="mermaid">
{exact_mmd.strip()}
        </pre>
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
</body>
</html>"""

with open(r"diagrams\TriSpecWave_Mamba_UQ_Exact.html", "w", encoding="utf-8") as f:
    f.write(html_content)

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        headless=True
    )
    context = browser.new_context(viewport={"width": 2400, "height": 3000}, device_scale_factor=2)
    page = context.new_page()
    page.goto(os.path.abspath(r"diagrams\TriSpecWave_Mamba_UQ_Exact.html"))
    page.wait_for_selector(".mermaid svg", timeout=15000)
    
    svg_element = page.locator(".mermaid svg")
    with open(r"diagrams\TriSpecWave_Mamba_UQ_Exact.svg", "w", encoding="utf-8") as f:
        f.write(svg_element.evaluate("el => el.outerHTML"))
        
    wrapper = page.locator("#chart-wrapper")
    box = wrapper.bounding_box()
    page.set_viewport_size({"width": int(box["x"] + box["width"] + 60), "height": int(box["y"] + box["height"] + 60)})
    wrapper.screenshot(path=r"diagrams\TriSpecWave_Mamba_UQ_Exact.png")
    browser.close()

print("Exact ASCII-structured diagram rendered successfully!")
