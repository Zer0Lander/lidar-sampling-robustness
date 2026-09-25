"""Deterministic projected-row sampling transforms.

Rows are computed in the original sensor frame. Apply the mask to all point-aligned arrays before augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ProjectionConfig:
    height: int = 64
    fov_up_deg: float = 3.0
    fov_down_deg: float = -25.0

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ValueError("height must be positive")
        if self.fov_up_deg <= self.fov_down_deg:
            raise ValueError("fov_up_deg must be greater than fov_down_deg")


def projected_rows(xyz: npt.NDArray[np.floating], config: ProjectionConfig = ProjectionConfig()) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.bool_]]:
    """Return clipped projection rows and the nominal-FOV membership mask.

    High elevations map to row 0 and low elevations to row H-1. Points outside the nominal FOV are clipped; ``in_fov`` reports their frequency.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
    if not np.issubdtype(xyz.dtype, np.floating):
        raise TypeError("xyz must use a floating-point dtype")

    horizontal_range = np.hypot(xyz[:, 0], xyz[:, 1])
    pitch = np.arctan2(xyz[:, 2], horizontal_range)
    fov_up = np.deg2rad(config.fov_up_deg)
    fov_down = np.deg2rad(config.fov_down_deg)
    in_fov = (pitch >= fov_down) & (pitch <= fov_up)

    normalized = 1.0 - (pitch - fov_down) / (fov_up - fov_down)
    rows = np.floor(normalized * config.height)
    rows = np.clip(rows, 0, config.height - 1).astype(np.int32)
    return rows, in_fov


def structured_mask(rows: npt.ArrayLike, *, stride: int, phase: int) -> npt.NDArray[np.bool_]:
    """Select one phase of a regular projected-row pattern."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not 0 <= phase < stride:
        raise ValueError(f"phase must satisfy 0 <= phase < stride ({stride})")
    row_array = np.asarray(rows)
    if row_array.ndim != 1 or not np.issubdtype(row_array.dtype, np.integer):
        raise TypeError("rows must be a one-dimensional integer array")
    return (row_array % stride) == phase


def exact_count_random_mask(point_count: int, retained_count: int, *, seed: int) -> npt.NDArray[np.bool_]:
    """Select exactly ``retained_count`` of ``point_count`` points."""
    if point_count < 0:
        raise ValueError("point_count must be non-negative")
    if not 0 <= retained_count <= point_count:
        raise ValueError("retained_count must be between zero and point_count")
    rng = np.random.default_rng(seed)
    selected = rng.choice(point_count, size=retained_count, replace=False)
    mask = np.zeros(point_count, dtype=np.bool_)
    mask[selected] = True
    return mask


def apply_mask(mask: npt.ArrayLike, *arrays: npt.NDArray[np.generic]) -> tuple[npt.NDArray[np.generic], ...]:
    """Apply one boolean mask to point-aligned arrays with validation."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 1 or mask_array.dtype != np.bool_:
        raise TypeError("mask must be a one-dimensional boolean array")
    for array in arrays:
        if len(array) != len(mask_array):
            raise ValueError("all arrays must have the same leading length as mask")
    return tuple(array[mask_array] for array in arrays)
