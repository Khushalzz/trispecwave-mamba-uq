import os
from playwright.sync_api import sync_playwright

horizontal_mmd = """flowchart LR
    classDef titleStyle fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef encStyle fill:#0f766e,stroke:#2dd4bf,stroke-width:2px,color:#ffffff;
    classDef fuseStyle fill:#b45309,stroke:#fcd34d,stroke-width:2px,color:#ffffff;
    classDef zoomStyle fill:#4338ca,stroke:#a5b4fc,stroke-width:2px,color:#ffffff;
    classDef outStyle fill:#065f46,stroke:#34d399,stroke-width:2px,color:#ffffff;
    classDef checkStyle fill:#991b1b,stroke:#fca5a5,stroke-width:2px,color:#ffffff;

    subgraph S1 ["1. SATELLITE INPUTS"]
        direction TB
        B10["<b>10m Sharp</b><br/>Blue, Green, Red, NIR"]:::inputStyle
        B20["<b>20m Medium</b><br/>Red Edge & SWIR"]:::inputStyle
        B60["<b>60m Blurry</b><br/>Atmospheric"]:::inputStyle
    end

    subgraph S2 ["2. FEATURE EXTRACTION"]
        direction TB
        WAVE["<b>Wavelet</b><br/>Extracts sharp outlines & edges"]:::encStyle
        MAMBA["<b>Mamba</b><br/>Learns color & wavelength physics"]:::encStyle
    end

    subgraph S3 ["3. GUIDED FUSION"]
        FUSE["<b>Smart Guided Fusion</b><br/>Uses 10m sharp outlines to<br/>place 20m & 60m blurry colors"]:::fuseStyle
    end

    subgraph S4 ["4. PROGRESSIVE ZOOM"]
        direction TB
        Z1["<b>Step 1:</b> Zoom to 5m"]:::zoomStyle
        Z2["<b>Step 2:</b> Zoom to 2.5m"]:::zoomStyle
    end

    subgraph S5 ["5. FINAL OUTPUTS (TriSpecWave-Mamba-UQ)"]
        direction TB
        SR["<b>12 Sharp Bands @ 2.5m</b><br/>Full multispectral resolution"]:::outStyle
        UQ["<b>AI Confidence Maps (UQ)</b><br/>Uncertainty score per band"]:::outStyle
    end

    subgraph S6 ["6. REALITY CHECK"]
        CHECK["<b>Physics Check</b><br/>Blurred output must match<br/>raw satellite photo"]:::checkStyle
    end

    B10 --> WAVE
    B20 --> MAMBA
    B60 --> MAMBA

    WAVE --> FUSE
    MAMBA --> FUSE

    FUSE --> Z1
    Z1 --> Z2

    Z2 --> SR
    Z2 --> UQ

    SR -.->|"Verify"| CHECK
"""

os.makedirs("diagrams", exist_ok=True)
with open(r"diagrams\Horizontal_Model_Architecture.mmd", "w", encoding="utf-8") as f:
    f.write(horizontal_mmd.strip())

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
            margin-bottom: 25px;
        }}
        .header h1 {{
            color: #38bdf8;
            font-size: 26px;
            margin: 0 0 8px 0;
            letter-spacing: 0.5px;
        }}
        .header p {{
            color: #94a3b8;
            font-size: 15px;
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
            <h1>TriSpecWave-Mamba-UQ Pipeline</h1>
            <p>End-to-End Flow: From 3 Satellite Blur Levels (10m / 20m / 60m) to 12 Razor-Sharp Bands @ 2.5m</p>
        </div>
        <pre class="mermaid">
{horizontal_mmd.strip()}
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

with open(r"diagrams\Horizontal_Model_Architecture.html", "w", encoding="utf-8") as f:
    f.write(html_content)

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        headless=True
    )
    context = browser.new_context(viewport={"width": 3200, "height": 1800}, device_scale_factor=2)
    page = context.new_page()
    page.goto(os.path.abspath(r"diagrams\Horizontal_Model_Architecture.html"))
    page.wait_for_selector(".mermaid svg", timeout=15000)
    
    svg_element = page.locator(".mermaid svg")
    with open(r"diagrams\Horizontal_Model_Architecture.svg", "w", encoding="utf-8") as f:
        f.write(svg_element.evaluate("el => el.outerHTML"))
        
    wrapper = page.locator("#chart-wrapper")
    box = wrapper.bounding_box()
    page.set_viewport_size({"width": int(box["x"] + box["width"] + 60), "height": int(box["y"] + box["height"] + 60)})
    wrapper.screenshot(path=r"diagrams\Horizontal_Model_Architecture.png")
    browser.close()

print("Rendered Horizontal_Model_Architecture successfully!")
