"""Native PyTorch training entry point."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.sen2naip_dataset import Sen2NAIPDataset
from losses.combined_loss import CombinedLoss
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ


def read_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read YAML configuration files") from exc
    with Path(path).open() as handle:
        return yaml.safe_load(handle) or {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


class SyntheticPairedDataset(Dataset):
    """Tiny deterministic smoke dataset used when no manifest is provided."""

    def __init__(self, count: int = 4, channels: int = 12, size: int = 16, scale: int = 4) -> None:
        self.items = []
        generator = torch.Generator().manual_seed(123)
        for _ in range(count):
            x = torch.rand(channels, size, size, generator=generator)
            y = F.interpolate(x.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False).squeeze(0)
            self.items.append({"input": x, "target": y, "mask": torch.ones(1, y.shape[-2], y.shape[-1])})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def build_model(model_cfg: dict[str, Any]) -> TriSpecWaveMambaUQ:
    return TriSpecWaveMambaUQ(**model_cfg)


def run_batch(model, criterion, batch, device, scaler, optimizer=None, grad_clip_norm: float = 1.0):
    x = batch["input"].to(device)
    y = batch["target"].to(device)
    mask = batch.get("mask")
    mask = None if mask is None else mask.to(device)
    train = optimizer is not None
    if train:
        optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp):
        mean, logvar = model(x)
        loss, parts = criterion(mean, logvar, y, native_input=x, mask=mask)
    if train:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    return float(loss.detach().cpu()), parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    model_cfg = read_yaml(args.model_config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    model = build_model(model_cfg).to(device)
    criterion = CombinedLoss()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)

    if args.dry_run:
        dataset = SyntheticPairedDataset(count=1, channels=model_cfg.get("in_channels", 12), size=16)
        loader = DataLoader(dataset, batch_size=1)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        loss, parts = run_batch(model, criterion, next(iter(loader)), device, scaler)
        print(json.dumps({"device": str(device), "loss": loss, "parts": {k: float(v) for k, v in parts.items()}}, indent=2))
        return

    dataset_cfg = read_yaml(args.dataset_config) if args.dataset_config else {}
    manifest = args.manifest or dataset_cfg.get("manifest")
    if manifest is None:
        raise SystemExit("Provide --manifest or --dataset-config, or use --dry-run for a smoke test")
    dataset = Sen2NAIPDataset(
        manifest,
        input_root=dataset_cfg.get("input_root"),
        target_root=dataset_cfg.get("target_root"),
        split=dataset_cfg.get("train_split", "train"),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 2)), shuffle=True, num_workers=int(config.get("num_workers", 2)), pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 2e-4)), weight_decay=float(config.get("weight_decay", 1e-5)))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    max_steps = args.max_steps
    global_step = 0
    for epoch in range(int(config.get("epochs", 50))):
        model.train()
        running = 0.0
        for batch in loader:
            loss, parts = run_batch(model, criterion, batch, device, scaler, optimizer, float(config.get("grad_clip_norm", 1.0)))
            running += loss
            global_step += 1
            if global_step % int(config.get("log_every", 20)) == 0:
                print(f"epoch={epoch + 1} step={global_step} loss={running / int(config.get('log_every', 20)):.6f}")
                running = 0.0
            if max_steps is not None and global_step >= max_steps:
                break
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "model_config": model_cfg}, checkpoint_dir / "last.pt")
        if max_steps is not None and global_step >= max_steps:
            break


if __name__ == "__main__":
    main()

