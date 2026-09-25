"""Minimal, dependency-light SemanticKITTI frame loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PointCloud:
    xyz: npt.NDArray[np.float32]
    reflectance: npt.NDArray[np.float32]
    semantic_labels: npt.NDArray[np.uint16] | None
    instance_labels: npt.NDArray[np.uint16] | None
    original_indices: npt.NDArray[np.int64]


def load_semantickitti_frame(scan_path: str | Path, label_path: str | Path | None = None) -> PointCloud:
    """Load one ``.bin`` scan and optional packed ``.label`` file."""
    scan_path = Path(scan_path)
    raw_scan = np.fromfile(scan_path, dtype=np.float32)
    if raw_scan.size % 4:
        raise ValueError(f"scan does not contain x/y/z/reflectance tuples: {scan_path}")
    points = raw_scan.reshape(-1, 4)
    count = len(points)

    semantic: npt.NDArray[np.uint16] | None = None
    instance: npt.NDArray[np.uint16] | None = None
    if label_path is not None:
        packed = np.fromfile(Path(label_path), dtype=np.uint32)
        if len(packed) != count:
            raise ValueError(f"point/label count mismatch: {count} points and {len(packed)} labels")
        semantic = (packed & np.uint32(0xFFFF)).astype(np.uint16)
        instance = (packed >> np.uint32(16)).astype(np.uint16)

    return PointCloud(
        xyz=points[:, :3],
        reflectance=points[:, 3],
        semantic_labels=semantic,
        instance_labels=instance,
        original_indices=np.arange(count, dtype=np.int64),
    )
