#!/usr/bin/env python3
"""Evaluate every checkpoint in a directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoints = sorted(Path(args.checkpoint_dir).glob("*.pt"))
    if not checkpoints:
        raise SystemExit(f"No .pt checkpoints found in {args.checkpoint_dir}")
    for checkpoint in checkpoints:
        command = [sys.executable, "evaluate.py", "--checkpoint", str(checkpoint), "--model-config", args.model_config]
        if args.manifest:
            command.extend(["--manifest", args.manifest])
        if args.dry_run:
            command.append("--dry-run")
        print(f"\n### {checkpoint}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

