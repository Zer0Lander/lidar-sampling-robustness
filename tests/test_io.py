import numpy as np
import pytest

from lidar_robustness.io import load_semantickitti_frame


def test_load_frame_decodes_labels(tmp_path) -> None:
    scan = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.7]], dtype=np.float32)
    semantic = np.array([10, 30], dtype=np.uint32)
    instance = np.array([2, 7], dtype=np.uint32)
    packed = semantic | (instance << 16)
    scan_path = tmp_path / "frame.bin"
    label_path = tmp_path / "frame.label"
    scan.tofile(scan_path)
    packed.tofile(label_path)

    frame = load_semantickitti_frame(scan_path, label_path)
    assert frame.semantic_labels is not None
    assert frame.instance_labels is not None
    np.testing.assert_array_equal(frame.xyz, scan[:, :3])
    np.testing.assert_array_equal(frame.reflectance, scan[:, 3])
    np.testing.assert_array_equal(frame.semantic_labels, semantic)
    np.testing.assert_array_equal(frame.instance_labels, instance)
    np.testing.assert_array_equal(frame.original_indices, np.arange(2))


def test_load_frame_rejects_misaligned_labels(tmp_path) -> None:
    scan_path = tmp_path / "frame.bin"
    label_path = tmp_path / "frame.label"
    np.zeros((2, 4), dtype=np.float32).tofile(scan_path)
    np.zeros(1, dtype=np.uint32).tofile(label_path)
    with pytest.raises(ValueError, match="point/label count mismatch"):
        load_semantickitti_frame(scan_path, label_path)
