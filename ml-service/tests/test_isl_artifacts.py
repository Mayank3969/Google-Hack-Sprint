"""Artifact graph/label checks use synthetic ZIPs and never import TensorFlow."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts.freeze_isl_artifacts import inspect_keras_archive, read_json, validate_label_maps


def maps():
    return ({f"label-{i}": i for i in range(50)}, {str(i): f"label-{i}" for i in range(50)})


def graph():
    return {"class_name": "Functional", "config": {
        "name": "test-model", "input_layers": ["input", 0, 0], "output_layers": ["prediction", 0, 0],
        "layers": [
            {"class_name": "InputLayer", "module": "keras.layers", "config": {"name": "input", "batch_shape": [None, 60, 660]}},
            {"class_name": "Dense", "module": "keras.layers", "config": {"name": "prediction", "units": 50, "activation": "softmax"}},
        ],
    }}


def archive(tmp_path, config, metadata=None):
    path = tmp_path / "fixture.keras"
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("config.json", json.dumps(config))
        output.writestr("model.weights.h5", b"not-real-weights-static-test-only")
        if metadata is not None:
            output.writestr("metadata.json", json.dumps(metadata))
    return path


def test_valid_maps_ordered_by_id():
    assert validate_label_maps(*maps()) == [f"label-{i}" for i in range(50)]


@pytest.mark.parametrize("bad_id", [True, 1.0, "1", -1, 50])
def test_label_map_rejects_invalid_ids(bad_id):
    forward, reverse = maps()
    forward["label-1"] = bad_id
    with pytest.raises(ValueError):
        validate_label_maps(forward, reverse)


@pytest.mark.parametrize("mutate", [
    lambda f, r: f.pop("label-49"),
    lambda f, r: r.update({"49": "label-0"}),
    lambda f, r: r.update({"050": r.pop("49")}),
    lambda f, r: f.update({"label-49": 48}),
])
def test_label_maps_require_exact_bijection(mutate):
    forward, reverse = maps()
    mutate(forward, reverse)
    with pytest.raises(ValueError):
        validate_label_maps(forward, reverse)


def test_json_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text('{"0": "one", "0": "two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_json(path)


def test_archive_signature_and_metadata(tmp_path):
    actual = inspect_keras_archive(archive(tmp_path, graph(), {"keras_version": "3.13.2", "tensorflow_version": "2.20.0"}))
    assert actual["input_batch_shape"] == [None, 60, 660]
    assert actual["output_probability_width"] == 50
    assert actual["keras_version"] == "3.13.2"
    assert actual["tensorflow_version"] == "2.20.0"


def test_archive_metadata_is_optional_and_singleton_endpoints_supported(tmp_path):
    config = graph()
    config["config"]["input_layers"] = [["input", 0, 0]]
    config["config"]["output_layers"] = [["prediction", 0, 0]]
    actual = inspect_keras_archive(archive(tmp_path, config))
    assert actual["keras_version"] is None
    assert actual["tensorflow_version"] is None


@pytest.mark.parametrize("shape", [[None, 60, 408], [None, 60, 330], [None, 59, 660], [1, 60, 660]])
def test_archive_rejects_wrong_shape(tmp_path, shape):
    config = graph()
    config["config"]["layers"][0]["config"]["batch_shape"] = shape
    with pytest.raises(ValueError, match="input batch shape"):
        inspect_keras_archive(archive(tmp_path, config))


@pytest.mark.parametrize("width,activation", [(49, "softmax"), (50, "linear"), (True, "softmax")])
def test_archive_rejects_non_probability_output(tmp_path, width, activation):
    config = graph()
    config["config"]["layers"][1]["config"].update(units=width, activation=activation)
    with pytest.raises(ValueError):
        inspect_keras_archive(archive(tmp_path, config))


def test_checks_actual_output_not_unrelated_dense(tmp_path):
    config = graph()
    decoy = copy.deepcopy(config["config"]["layers"][1])
    decoy["config"]["name"] = "decoy-50-way"
    config["config"]["layers"].append(decoy)
    config["config"]["layers"][1]["config"]["units"] = 49
    with pytest.raises(ValueError, match="output probability width"):
        inspect_keras_archive(archive(tmp_path, config))


def test_missing_weights_rejected(tmp_path):
    path = tmp_path / "missing.keras"
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("config.json", json.dumps(graph()))
    with pytest.raises(ValueError, match="model.weights.h5"):
        inspect_keras_archive(path)


def test_static_import_does_not_load_tensorflow():
    root = Path(__file__).resolve().parents[1]
    check = subprocess.run([sys.executable, "-c",
        "import sys; from scripts import freeze_isl_artifacts, evaluate_isl; "
        "assert 'tensorflow' not in sys.modules; assert 'keras' not in sys.modules"],
        cwd=root, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
