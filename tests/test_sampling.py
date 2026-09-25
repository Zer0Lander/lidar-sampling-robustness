import numpy as np

from lidar_robustness.sampling import (
    apply_mask,
    exact_count_random_mask,
    projected_rows,
    structured_mask,
)


def test_stride_one_is_identity() -> None:
    rows = np.arange(64, dtype=np.int32)
    assert structured_mask(rows, stride=1, phase=0).all()


def test_phases_partition_points() -> None:
    rows = np.repeat(np.arange(64, dtype=np.int32), 3)
    for stride in (2, 3, 4):
        masks = [structured_mask(rows, stride=stride, phase=p) for p in range(stride)]
        assert np.stack(masks).sum(axis=0).tolist() == [1] * len(rows)


def test_random_mask_is_exact_and_deterministic() -> None:
    first = exact_count_random_mask(101, 37, seed=123)
    second = exact_count_random_mask(101, 37, seed=123)
    third = exact_count_random_mask(101, 37, seed=124)
    assert first.sum() == 37
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_mask_preserves_alignment() -> None:
    indices = np.arange(20, dtype=np.int64)
    xyz = np.column_stack((indices, indices + 100, indices + 200)).astype(np.float32)
    labels = (indices + 1000).astype(np.uint16)
    mask = (indices % 3) == 1
    masked_xyz, masked_labels, masked_indices = apply_mask(mask, xyz, labels, indices)
    np.testing.assert_array_equal(masked_xyz, xyz[mask])
    np.testing.assert_array_equal(masked_labels, labels[mask])
    np.testing.assert_array_equal(masked_indices, indices[mask])


def test_projection_direction_and_clipping() -> None:
    angles = np.deg2rad(np.array([3.0, -11.0, -25.0, 10.0, -30.0]))
    xyz = np.column_stack((np.cos(angles), np.zeros_like(angles), np.sin(angles))).astype(np.float32)
    rows, in_fov = projected_rows(xyz)
    np.testing.assert_array_equal(rows, np.array([0, 32, 63, 0, 63], dtype=np.int32))
    np.testing.assert_array_equal(in_fov, np.array([True, True, True, False, False]))
