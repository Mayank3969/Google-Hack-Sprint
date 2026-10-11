from __future__ import annotations

import pickle

import numpy as np
import pytest

from service.inference import MAX_FRAMES, InferenceError, predict_landmarks


class KnownClassifier:
    def __init__(self, label: str = "wave"):
        self.label = label
        self.inputs: list[np.ndarray] = []

    def predict(self, sample):
        self.inputs.append(sample)
        return np.array([self.label])


@pytest.mark.parametrize("frames", [1, MAX_FRAMES, MAX_FRAMES + 1])
def test_predict_pads_exact_and_truncates(frames):
    classifier = KnownClassifier()
    landmarks = np.ones((frames, 21, 3), dtype=np.float32)

    assert predict_landmarks(landmarks, classifier=classifier) == ["wave"]
    fixed = classifier.inputs[0]
    assert fixed.shape == (1, MAX_FRAMES * 21 * 3)
    assert np.all(fixed[0, : 21 * 3] == 1)
    if frames < MAX_FRAMES:
        assert np.all(fixed[0, frames * 21 * 3 :] == 0)
    else:
        assert np.all(fixed == 1)


def test_predict_accepts_all_zero_landmarks():
    classifier = KnownClassifier("still")
    assert predict_landmarks(np.zeros((2, 21, 3)), classifier=classifier) == ["still"]


@pytest.mark.parametrize(
    "landmarks, message",
    [
        (np.zeros((21, 3)), "shape"),
        (np.zeros((1, 20, 3)), "shape"),
        ([[["x", 0, 0]] * 21], "numeric"),
        (np.full((1, 21, 3), np.nan), "finite"),
        (np.full((1, 21, 3), np.inf), "finite"),
    ],
)
def test_predict_rejects_malformed_landmarks(landmarks, message):
    with pytest.raises(ValueError, match=message):
        predict_landmarks(landmarks, classifier=KnownClassifier())


def test_predict_loads_known_synthetic_classifier_artifact(tmp_path):
    artifact = tmp_path / "classifier.pkl"
    with artifact.open("wb") as handle:
        pickle.dump(KnownClassifier("artifact-label"), handle)

    assert predict_landmarks(np.zeros((MAX_FRAMES, 21, 3)), model_path=artifact) == [
        "artifact-label"
    ]


def test_missing_model_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="classifier model does not exist"):
        predict_landmarks(np.zeros((1, 21, 3)), model_path=tmp_path / "missing.pkl")


@pytest.mark.parametrize("result", [[], ["a", "b"], [1], None])
def test_malformed_classifier_output_is_clear(result):
    class BadClassifier:
        def predict(self, sample):
            return result

    with pytest.raises(InferenceError, match="exactly one predicted label|string label"):
        predict_landmarks(np.zeros((1, 21, 3)), classifier=BadClassifier())
