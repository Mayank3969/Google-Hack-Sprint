"""M3 tests for fixed-length (60, 110, 3) samples."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import build_dataset_module as builder


def video(root, label="pay", name="one.mp4"):
    path = root / label / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def fake_extractor(path, **kwargs):
    return np.full((2, *builder.FEATURE_SHAPE), 0.25, dtype=np.float32)


@pytest.mark.parametrize("frames", [1, 59, 60, 61])
def test_padding_and_truncation(frames):
    original = np.zeros((frames, *builder.FEATURE_SHAPE), dtype=np.float32)
    original[:, 0, 0] = np.arange(frames)
    result = builder.pad_or_truncate(original)
    assert result.shape == (60, 110, 3)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:min(frames, 60), 0, 0], np.arange(min(frames, 60)))
    assert not result[min(frames, 60):].any()


def test_deterministic_build_and_metadata(tmp_path):
    raw = tmp_path / "raw"
    video(raw, "pay", "b.mp4")
    video(raw, "pay", "a.mp4")
    video(raw, "薬", "z.mp4")
    calls = []
    def extract(path, **kwargs):
        calls.append(path.name)
        return np.full((1, *builder.FEATURE_SHAPE), len(calls), dtype=np.float32)
    output = tmp_path / "out" / "dataset.npz"
    X, y, metadata = builder.build_dataset(raw, output, extractor=extract)
    assert calls == ["a.mp4", "b.mp4", "z.mp4"]
    assert X.shape == (3, 60, 110, 3)
    assert y.tolist() == ["pay", "pay", "薬"]
    saved = np.load(output, allow_pickle=False)
    assert saved["X"].shape == X.shape and saved["y"].dtype.kind == "U"
    assert json.loads(builder.metadata_path(output).read_text(encoding="utf-8")) == metadata
    assert metadata["feature_shape"] == [110, 3]
    assert metadata["hand_slots"] == ["left", "right"]
    assert metadata["face_points"] == 68
    assert metadata["face_feature_version"] == "compact-v1"


def test_rejects_wrong_shape_and_empty_label(tmp_path):
    raw = tmp_path / "raw"
    video(raw)
    with pytest.raises(ValueError, match="110"):
        builder.build_dataset(raw, tmp_path / "out.npz", extractor=lambda path, **k: np.zeros((1, 21, 3)))
    empty = tmp_path / "empty"
    (empty / "label").mkdir(parents=True)
    with pytest.raises(ValueError, match="no usable"):
        builder.build_dataset(empty, tmp_path / "empty.npz", extractor=fake_extractor)


def test_default_extractor_requires_face_model(tmp_path):
    raw = tmp_path / "raw"
    video(raw)
    with pytest.raises(ValueError, match="face-landmarker model"):
        builder.build_dataset(raw, tmp_path / "out.npz")
