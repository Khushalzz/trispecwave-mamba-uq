import os
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_sen2neon(target_dir="/mnt/pune/sen2neon"):
    os.makedirs(target_dir, exist_ok=True)
    
    logger.info(f"Downloading SEN2NEON to {target_dir}...")
    snapshot_download(
        repo_id="isp-uv-es/SEN2NEON",
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=["metadata.csv", "s2_l2a_10m/*", "neon_2.5m_linearized/*"]
    )
    logger.info("Download complete!")

if __name__ == "__main__":
    download_sen2neon()
