#!/usr/bin/env python3
"""Inspect one labeled SemanticKITTI frame under all sampling conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from lidar_robustness.io import load_semantickitti_frame
from lidar_robustness.sampling import exact_count_random_mask, projected_rows, structured_mask


def mask_hash(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([len(mask)], dtype=np.int64).tobytes())
    digest.update(np.packbits(mask, bitorder="little").tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True, help="Path to a .bin scan")
    parser.add_argument("--label", type=Path, required=True, help="Path to its .label file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def record(name: str, mask: np.ndarray, *, total: int) -> dict[str, object]:
    retained = int(mask.sum())
    return {
        "condition": name,
        "point_count": retained,
        "retained_percent": 100.0 * retained / total if total else 0.0,
        "mask_sha256": mask_hash(mask),
    }


def save_row_occupancy(output_path: Path, rows: np.ndarray, *, height: int = 64) -> None:
    counts = np.bincount(rows, minlength=height)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row", "point_count", "point_percent"))
        for row, count in enumerate(counts):
            percent = 100.0 * int(count) / len(rows) if len(rows) else 0.0
            writer.writerow((row, int(count), percent))


def main() -> None:
    args = parse_args()
    frame = load_semantickitti_frame(args.scan, args.label)
    rows, in_fov = projected_rows(frame.xyz)
    point_count = len(rows)
    full = np.ones(point_count, dtype=np.bool_)

    records: list[dict[str, object]] = [record("E64", full, total=point_count)]
    for name, stride in (("E32", 2), ("E16", 4), ("E21", 3)):
        for phase in range(stride):
            condition = f"{name}_phase{phase}"
            mask = structured_mask(rows, stride=stride, phase=phase)
            records.append(record(condition, mask, total=point_count))
            if name in {"E32", "E16"}:
                random_name = name.replace("E", "R")
                for seed in args.random_seeds:
                    random_condition = f"{random_name}_phase{phase}_seed{seed}"
                    random_mask = exact_count_random_mask(point_count, int(mask.sum()), seed=seed)
                    records.append(record(random_condition, random_mask, total=point_count))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sampling_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "scan": str(args.scan.resolve()),
        "label": str(args.label.resolve()),
        "point_count": point_count,
        "outside_nominal_fov_count": int((~in_fov).sum()),
        "outside_nominal_fov_percent": 100.0 * float((~in_fov).mean()) if point_count else 0.0,
        "random_seeds": args.random_seeds,
    }
    (args.output_dir / "frame_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_row_occupancy(args.output_dir / "row_occupancy.csv", rows)

    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output_dir / 'sampling_stats.csv'}")
    print(f"wrote {args.output_dir / 'row_occupancy.csv'}")


if __name__ == "__main__":
    main()
