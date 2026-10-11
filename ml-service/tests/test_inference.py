"""M5 tests for checkpoint loading and mobile-shaped inference."""

from pathlib import Path
import importlib

import numpy as np
import pytest


torch = pytest.importorskip("torch")
trainer = importlib.import_module("scripts.04_train_bigru")
inference = importlib.import_module("service.inference")


def make_checkpoint(path: Path) -> Path:
    config = trainer.TrainConfig(hidden_size=8, layers=1, dropout=0.0)
    model = trainer.build_model(3, config)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "input_size": config.input_size,
            "hidden_size": config.hidden_size,
            "layers": config.layers,
            "dropout": config.dropout,
        },
        "label_names": ["alpha", "beta", "gamma"],
        "feature_shape": (110, 3),
        "max_frames": 60,
        "preprocessing_version": inference.PREPROCESSING_VERSION,
    }
    torch.save(checkpoint, path)
    return path


def test_temporal_preprocessing_matches_m4():
    sample = np.zeros((12, 110, 3), dtype=np.float32)
    sample[:8, 0, 0] = np.arange(8, dtype=np.float32)
    actual = inference.temporal_normalize(sample)
    expected = trainer.temporal_normalize(sample)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (60, 110, 3)


def test_predict_details_loads_checkpoint_and_returns_word(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "bigru_best.pt")
    engine = inference.InferenceEngine(checkpoint, device="cpu")
    sample = np.zeros((12, 110, 3), dtype=np.float32)
    sample[:, 0, 0] = 0.25

    result = engine.predict_details(sample, top_k=2).as_dict()

    assert result["label"] in {"alpha", "beta", "gamma"}
    assert 0 <= result["class_id"] < 3
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_k"]) == 2
    assert result["top_k"][0]["label"] == result["label"]


@pytest.mark.parametrize(
    "sample",
    [
        np.zeros((0, 110, 3), dtype=np.float32),
        np.zeros((61, 110, 3), dtype=np.float32),
        np.zeros((10, 42, 3), dtype=np.float32),
        np.zeros((10, 110, 2), dtype=np.float32),
    ],
)
def test_predict_rejects_invalid_mobile_shapes(tmp_path, sample):
    engine = inference.InferenceEngine(make_checkpoint(tmp_path / "bigru_best.pt"), device="cpu")
    with pytest.raises(ValueError):
        engine.predict_details(sample)


def test_predict_rejects_incompatible_checkpoint(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "bigru_best.pt")
    data = torch.load(checkpoint, weights_only=False)
    data["preprocessing_version"] = "wrong-version"
    torch.save(data, checkpoint)
    with pytest.raises(ValueError, match="preprocessing version"):
        inference.InferenceEngine(checkpoint, device="cpu")
