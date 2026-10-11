#!/usr/bin/env python3
"""Build a simple paired-raster manifest from matching filenames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="*.tif")
    parser.add_argument("--split", default="adaptation")
    args = parser.parse_args()
    input_dir, target_dir = Path(args.input_dir), Path(args.target_dir)
    targets = {path.stem: path for path in target_dir.glob(args.pattern)}
    rows = []
    for input_path in sorted(input_dir.glob(args.pattern)):
        target_path = targets.get(input_path.stem)
        if target_path is not None:
            rows.append({"input": str(input_path), "target": str(target_path), "split": args.split})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input", "target", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pairs to {output}")


if __name__ == "__main__":
    main()

