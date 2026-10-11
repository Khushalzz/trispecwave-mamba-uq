import os
import rasterio
from rasterio.windows import Window

import numpy as np
from PIL import Image

def get_target():
    scene_dir = "/mnt/pune/raw/S2A_MSIL2A_20240304T052721_R105_T43QCA_20240304T101311"
    lat, lon = 18.4388, 73.7728
    
    # We want a 128x128 patch at 10m resolution (so 1280m x 1280m)
    patch_size = 128
    
    # B2 (Blue), B3 (Green), B4 (Red) are 10m bands
    rgb_files = [f"{scene_dir}/B4.tif", f"{scene_dir}/B3.tif", f"{scene_dir}/B2.tif"]
    
    rgb_arrays = []
    
    with rasterio.open(rgb_files[0]) as src:
        # Convert lat/lon to UTM
        from rasterio.warp import transform
        xs, ys = transform("epsg:4326", src.crs, [lon], [lat])
        utm_x, utm_y = xs[0], ys[0]
        
        # Get pixel row/col
        row, col = src.index(utm_x, utm_y)
        
        # Calculate window around this pixel
        start_row = max(0, row - patch_size // 2)
        start_col = max(0, col - patch_size // 2)
        
        window = Window(start_col, start_row, patch_size, patch_size)
    
    for f in rgb_files:
        with rasterio.open(f) as src:
            data = src.read(1, window=window)
            rgb_arrays.append(data)
            
    rgb = np.stack(rgb_arrays, axis=-1).astype(np.float32)
    
    # Normalize (typically Sentinel-2 L2A reflectance is in 0-10000 range)
    # The uploaded image looks a bit bright. Let's use 2000.0 or percentile.
    # To match standard visualization, let's auto-scale or just divide by 2500.0
    rgb = np.clip(rgb / 2500.0, 0, 1) * 255.0
    rgb = rgb.astype(np.uint8)
    
    img = Image.fromarray(rgb)
    out_path = "/mnt/pune/target_pune.png"
    img.save(out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    get_target()
