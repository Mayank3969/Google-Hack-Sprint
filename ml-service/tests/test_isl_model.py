"""B1 unit and REAL-ARTIFACT tests.

INTEGRATION_WARNING: random input below tests serialization/inference only.
Person A must also pass the independent golden-fixture release check; these
tests do not prove label accuracy or compact-face/hand ordering.
"""

import json

import numpy as np
import pytest

from service.isl_model import DEFAULT_MANIFEST_PATH, DEFAULT_MODEL_PATH, ISLKerasEngine, _load_labels


@pytest.mark.parametrize("case", ["missing", "extra", "blank", "numeric", "duplicate"])
def test_label_schema(tmp_path, case):
    labels = {str(i): f"sign{i}" for i in range(50)}
    if case == "missing":
        del labels["0"]
    elif case == "extra":
        labels["50"] = "extra"
    elif case == "blank":
        labels["0"] = " "
    elif case == "numeric":
        labels["0"] = 123
    else:
        labels["0"] = labels["1"]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_labels(path)


@pytest.mark.real_model
@pytest.mark.parametrize("name", ["best_isl_model_v2.keras", "isl_1dcnn_transformer_v2.keras"])
def test_committed_models_load_and_batch_matches_single(name):
    # Deliberately fail, not skip, if the selected real-artifact test lacks TF.
    engine = ISLKerasEngine(DEFAULT_MODEL_PATH.with_name(name))
    data = np.random.default_rng(42).normal(0, 0.01, (2, 60, 660)).astype(np.float32)
    batched = engine.predict_probabilities(data)
    individual = np.concatenate([engine.predict_probabilities(row[None]) for row in data])
    assert batched.shape == (2, 50)
    np.testing.assert_allclose(batched.sum(axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(batched, individual, atol=1e-5, rtol=1e-4)
    assert len(engine.labels) == 50
    assert len(engine.model_sha256) == 64


@pytest.mark.real_model
def test_model_failure_and_feature_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        ISLKerasEngine(tmp_path / "absent.keras")
    engine = ISLKerasEngine()
    for data in [np.zeros((1, 60, 330)), np.zeros((0, 60, 660)), np.full((1, 60, 660), np.nan)]:
        with pytest.raises(ValueError):
            engine.predict_probabilities(data)


@pytest.mark.real_model
def test_manifest_mismatch_is_not_silently_loaded(tmp_path):
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["labels_sha256"] = "wrong"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="label checksum"):
        ISLKerasEngine(manifest_path=path)


@pytest.mark.real_model
@pytest.mark.parametrize("input_shape,output_shape", [((None, 60, 330), (None, 50)), ((None, 60, 660), (None, 49))])
def test_wrong_model_shapes_fail(monkeypatch, input_shape, output_shape):
    import keras
    class WrongShape:
        pass
    wrong = WrongShape()
    wrong.input_shape, wrong.output_shape = input_shape, output_shape
    monkeypatch.setattr(keras.models, "load_model", lambda *a, **kw: wrong)
    with pytest.raises(ValueError, match="ISL model"):
        ISLKerasEngine()
