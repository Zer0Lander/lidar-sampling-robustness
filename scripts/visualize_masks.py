#!/usr/bin/env python3
"""Visualize projected-row sampling conditions for one SemanticKITTI frame."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from lidar_robustness.io import load_semantickitti_frame
from lidar_robustness.sampling import exact_count_random_mask, projected_rows, structured_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def draw_topdown(axis, xyz: np.ndarray, mask: np.ndarray, title: str) -> None:
    removed = xyz[~mask]
    kept = xyz[mask]
    axis.scatter(removed[:, 0], removed[:, 1], s=0.08, color="0.86", rasterized=True)
    axis.scatter(kept[:, 0], kept[:, 1], s=0.12, color="#1764ab", rasterized=True)
    axis.set_title(f"{title}\n{mask.sum():,} points, {100 * mask.mean():.1f}%")
    axis.set_aspect("equal")
    axis.set_xlim(-60, 60)
    axis.set_ylim(-60, 60)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")


def draw_range(axis, azimuth: np.ndarray, rows: np.ndarray, mask: np.ndarray, title: str) -> None:
    axis.scatter(azimuth[~mask], rows[~mask], s=0.08, color="0.86", rasterized=True)
    axis.scatter(azimuth[mask], rows[mask], s=0.12, color="#1764ab", rasterized=True)
    axis.set_title(f"{title}\n{mask.sum():,} points, {100 * mask.mean():.1f}%")
    axis.set_xlim(-180, 180)
    axis.set_ylim(63.5, -0.5)
    axis.set_xlabel("azimuth (degrees)")
    axis.set_ylabel("projected row")


def save_structured_phases(output_path: Path, xyz: np.ndarray, rows: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    conditions = [("E32", 2), ("E16", 4), ("E21", 3)]
    figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for row_index, (name, stride) in enumerate(conditions):
        for phase in range(4):
            axis = axes[row_index, phase]
            if phase >= stride:
                axis.axis("off")
                continue
            mask = structured_mask(rows, stride=stride, phase=phase)
            draw_topdown(axis, xyz, mask, f"{name} phase {phase}")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_structured_range_phases(output_path: Path, xyz: np.ndarray, rows: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    azimuth = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    conditions = [("E32", 2), ("E16", 4), ("E21", 3)]
    figure, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    for row_index, (name, stride) in enumerate(conditions):
        for phase in range(4):
            axis = axes[row_index, phase]
            if phase >= stride:
                axis.axis("off")
                continue
            mask = structured_mask(rows, stride=stride, phase=phase)
            draw_range(axis, azimuth, rows, mask, f"{name} phase {phase}")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_random_comparison(output_path: Path, xyz: np.ndarray, rows: np.ndarray, *, name: str, stride: int) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(stride, 2, figsize=(10, 5 * stride), constrained_layout=True)
    if stride == 1:
        axes = np.asarray([axes])
    for phase in range(stride):
        structured = structured_mask(rows, stride=stride, phase=phase)
        random = exact_count_random_mask(len(rows), int(structured.sum()), seed=0)
        draw_topdown(axes[phase, 0], xyz, structured, f"{name} phase {phase}")
        draw_topdown(axes[phase, 1], xyz, random, f"R{name[1:]} matched, seed 0")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_range_random_comparison(output_path: Path, xyz: np.ndarray, rows: np.ndarray, *, name: str, stride: int) -> None:
    import matplotlib.pyplot as plt

    azimuth = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    figure, axes = plt.subplots(stride, 2, figsize=(12, 3 * stride), constrained_layout=True)
    if stride == 1:
        axes = np.asarray([axes])
    for phase in range(stride):
        structured = structured_mask(rows, stride=stride, phase=phase)
        random = exact_count_random_mask(len(rows), int(structured.sum()), seed=0)
        draw_range(axes[phase, 0], azimuth, rows, structured, f"{name} phase {phase}")
        draw_range(axes[phase, 1], azimuth, rows, random, f"R{name[1:]} matched, seed 0")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_projection_rows(output_path: Path, xyz: np.ndarray, rows: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    azimuth = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    scatter = axis.scatter(azimuth, rows, c=rows, s=0.2, cmap="turbo", rasterized=True)
    axis.set_xlabel("azimuth (degrees)")
    axis.set_ylabel("projected row")
    axis.set_ylim(63.5, -0.5)
    axis.set_title("Full scan projected into 64 vertical rows")
    figure.colorbar(scatter, ax=axis, label="projected row")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_row_occupancy(output_path: Path, rows: np.ndarray, *, height: int = 64) -> None:
    import matplotlib.pyplot as plt

    counts = np.bincount(rows, minlength=height)
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.bar(np.arange(height), counts, width=0.85)
    axis.set_xlabel("projected row")
    axis.set_ylabel("point count")
    axis.set_title("Projected-row occupancy")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/lidar-robustness-matplotlib")
    frame = load_semantickitti_frame(args.scan, args.label)
    rows, _ = projected_rows(frame.xyz)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_projection_rows(args.output_dir / "projection_rows.png", frame.xyz, rows)
    save_row_occupancy(args.output_dir / "row_occupancy.png", rows)
    save_structured_phases(args.output_dir / "structured_phases.png", frame.xyz, rows)
    save_structured_range_phases(args.output_dir / "structured_range_phases.png", frame.xyz, rows)
    save_random_comparison(args.output_dir / "e32_random_comparison.png", frame.xyz, rows, name="E32", stride=2)
    save_random_comparison(args.output_dir / "e16_random_comparison.png", frame.xyz, rows, name="E16", stride=4)
    save_range_random_comparison(args.output_dir / "e32_range_random_comparison.png", frame.xyz, rows, name="E32", stride=2)
    save_range_random_comparison(args.output_dir / "e16_range_random_comparison.png", frame.xyz, rows, name="E16", stride=4)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
