"""Evaluate a checkpoint on a manifest or a synthetic smoke sample."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from data.sen2naip_dataset import Sen2NAIPDataset
from metrics.spatial_metrics import lpips_proxy, psnr, ssim
from metrics.spectral_metrics import band_correlation, ergas, spectral_angle_mapper_metric
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from train import SyntheticPairedDataset, read_yaml, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model_cfg = read_yaml(args.model_config)
    device = resolve_device(args.device)
    model = TriSpecWaveMambaUQ(**model_cfg).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=False)
    if args.dry_run:
        dataset = SyntheticPairedDataset(count=1, channels=model_cfg.get("in_channels", 12), size=16)
    else:
        dataset_cfg = read_yaml(args.dataset_config) if args.dataset_config else {}
        manifest = args.manifest or dataset_cfg.get("manifest")
        if not manifest:
            raise SystemExit("Provide --manifest or --dataset-config, or use --dry-run")
        dataset = Sen2NAIPDataset(manifest, input_root=dataset_cfg.get("input_root"), target_root=dataset_cfg.get("target_root"), split=dataset_cfg.get("val_split", "val"))
    loader = DataLoader(dataset, batch_size=1)
    values = []
    with torch.no_grad():
        for batch in loader:
            mean, _ = model(batch["input"].to(device))
            target = batch["target"].to(device)
            values.append({
                "psnr": float(psnr(mean, target).cpu()),
                "ssim": float(ssim(mean, target).cpu()),
                "lpips_proxy": float(lpips_proxy(mean, target).cpu()),
                "sam": float(spectral_angle_mapper_metric(mean, target).cpu()),
                "ergas": float(ergas(mean, target).cpu()),
                "band_correlation": float(band_correlation(mean, target).mean().cpu()),
            })
    keys = values[0].keys() if values else []
    result = {key: sum(item[key] for item in values) / len(values) for key in keys}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

