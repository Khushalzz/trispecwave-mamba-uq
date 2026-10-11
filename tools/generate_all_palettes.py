import os
import shutil
import pathlib

# Matplotlib sandbox compatibility fix
pathlib.Path.mkdir = lambda self, mode=511, parents=False, exist_ok=False: None

from PIL import Image
import numpy as np
import matplotlib
import matplotlib.cm as cm
from scipy.spatial import cKDTree

def process():
    input_path = 'uncertainty_map.png'
    orig_img = Image.open(input_path).convert('RGB')
    arr = np.array(orig_img)

    # 1. High-resolution LUT for the original RdYlGn_r colormap
    N = 4096
    t_vals = np.linspace(0, 1, N)
    rdylgn_lut = (matplotlib.colormaps['RdYlGn_r'](t_vals)[:, :3] * 255).astype(np.float32)
    tree = cKDTree(rdylgn_lut)

    # 2. Extract and invert the heatmap to normalized scalar values t in [0, 1]
    hm_rgb = arr[84:616, 52:584].astype(np.float32)
    flat_hm = hm_rgb.reshape(-1, 3)
    _, indices = tree.query(flat_hm)
    t_hm = t_vals[indices].reshape((532, 532))

    # 3. Colorbar coordinates & parameters
    cbar_y = np.arange(82, 617)
    t_cbar = (616 - cbar_y) / (616 - 82)

    # Colormaps to generate
    palettes = ['viridis', 'plasma', 'inferno', 'turbo', 'magma', 'coolwarm', 'cividis', 'jet']
    
    artifact_dir = r"C:\Users\Ketamania\.gemini\antigravity\brain\5f931123-5a0b-41bc-9671-9d9c78d746d6"
    downloads_dir = r"C:\Users\Ketamania\Downloads"

    os.makedirs(artifact_dir, exist_ok=True)

    generated = {}
    for name in palettes:
        cmap = matplotlib.colormaps[name]
        new_hm_rgb = (cmap(t_hm)[:, :, :3] * 255).round().astype(np.uint8)
        new_cbar_rgb = (cmap(t_cbar)[:, :3] * 255).round().astype(np.uint8)
        new_cbar_block = np.repeat(new_cbar_rgb[:, np.newaxis, :], 634 - 608, axis=1)

        out_arr = arr.copy()
        out_arr[84:616, 52:584] = new_hm_rgb
        out_arr[82:617, 608:634] = new_cbar_block

        out_img = Image.fromarray(out_arr)
        
        # Save locally in workspace
        local_filename = f"uncertainty_map_{name}.png"
        out_img.save(local_filename)

        # Save to artifact dir for markdown visualization
        artifact_path = os.path.join(artifact_dir, local_filename)
        out_img.save(artifact_path)

        generated[name] = local_filename
        print(f"Generated {name} palette.")

    # Save original to artifact dir as well for side-by-side comparison
    orig_img.save(os.path.join(artifact_dir, "uncertainty_map_original.png"))

    print("All palettes generated successfully!")

if __name__ == '__main__':
    process()
