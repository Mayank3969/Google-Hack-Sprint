from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import dataset_builder


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "02_build_dataset.py"
    spec = importlib.util.spec_from_file_location("m3_cli", path)
    cli = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cli)
    return cli


def test_build_dataset_is_sorted_padded_and_metadata_is_saved(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "z-sign").mkdir(parents=True)
    (raw / "a-sign").mkdir()
    for name in ("z.mp4", "a.MOV"):
        (raw / "z-sign" / name).write_bytes(b"video")
    (raw / "a-sign" / "b.mp4").write_bytes(b"video")
    (raw / "a-sign" / "notes.txt").write_text("ignore")

    def fake_extract(path):
        frames = 61 if Path(path).name == "z.mp4" else 2
        return np.ones((frames, 21, 3), dtype=np.float64) * len(Path(path).name)

    monkeypatch.setattr(dataset_builder, "extract_landmarks", fake_extract)
    output = tmp_path / "model" / "dataset.npz"
    metadata = dataset_builder.build_dataset(raw, output)
    data = np.load(output)
    assert data["X"].shape == (3, 60, 21, 3)
    assert data["y"].tolist() == ["a-sign", "z-sign", "z-sign"]
    assert np.all(data["X"][0, 2:] == 0)
    assert metadata["per_label_counts"] == {"a-sign": 1, "z-sign": 2}
    assert json.loads(output.with_name("dataset_metadata.json").read_text())["max_frames"] == 60


@pytest.mark.parametrize("setup, message", [("missing", "does not exist"), ("empty", "no label folders")])
def test_missing_or_empty_raw_directory_fails_clearly(tmp_path, setup, message):
    raw = tmp_path / "raw"
    if setup == "empty":
        raw.mkdir()
    with pytest.raises((FileNotFoundError, ValueError), match=message):
        dataset_builder.build_dataset(raw, tmp_path / "model" / "dataset.npz")


def test_empty_label_and_invalid_extraction_fail(tmp_path):
    raw = tmp_path / "raw"
    (raw / "hello").mkdir(parents=True)
    with pytest.raises(ValueError, match="no video files"):
        dataset_builder.build_dataset(raw, tmp_path / "model" / "dataset.npz")
    (raw / "hello" / "sample.mp4").write_bytes(b"video")
    with pytest.raises(ValueError, match="invalid data"):
        dataset_builder.build_dataset(raw, tmp_path / "model" / "dataset.npz", extractor=lambda _: np.zeros((0, 21, 3)))


def test_cli_wrapper_imports_from_repository_root():
    assert callable(_load_cli().main)
