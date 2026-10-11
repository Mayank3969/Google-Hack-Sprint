from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from scripts.train_classifier import train_classifier


def _dataset(path, *, samples_per_label=5, shape=(60, 21, 3)):
    rng = np.random.default_rng(7)
    x = []
    y = []
    for label, offset in (("open", 0.0), ("closed", 10.0)):
        x.extend(rng.normal(offset, 0.01, (samples_per_label, *shape)))
        y.extend([label] * samples_per_label)
    np.savez(path, X=np.asarray(x, dtype=np.float32), y=np.asarray(y))


def test_train_classifier_saves_deterministic_artifact_and_metadata(tmp_path):
    dataset = tmp_path / "dataset.npz"
    output = tmp_path / "model" / "classifier.pkl"
    _dataset(dataset)
    first = train_classifier(dataset, output)
    second = train_classifier(dataset, output)
    assert first["validation_accuracy"] == second["validation_accuracy"] == 1.0
    assert first["labels"] == ["closed", "open"]
    assert output.is_file()
    assert pickle.load(output.open("rb")).random_state == 42
    saved = json.loads((output.parent / "classifier_metadata.json").read_text())
    assert saved["max_frames"] == 60
    assert saved["coordinate_format"]
    assert saved["code_version"]
    assert saved["training_date"]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda x, y: (x[:, :59], y), "shape"),
        (lambda x, y: (x, y[:-1]), "1-D array"),
    ],
)
def test_invalid_dataset_boundaries_fail(tmp_path, mutator, message):
    x = np.zeros((4, 60, 21, 3), dtype=np.float32)
    y = np.array(["a", "a", "b", "b"])
    x, y = mutator(x, y)
    dataset = tmp_path / "dataset.npz"
    np.savez(dataset, X=x, y=y)
    with pytest.raises(ValueError, match=message):
        train_classifier(dataset, tmp_path / "classifier.pkl")


def test_stratified_split_requires_two_samples_per_label(tmp_path):
    dataset = tmp_path / "dataset.npz"
    np.savez(dataset, X=np.zeros((3, 60, 21, 3)), y=np.array(["a", "a", "b"]))
    with pytest.raises(ValueError, match="at least 2 samples"):
        train_classifier(dataset, tmp_path / "classifier.pkl")
