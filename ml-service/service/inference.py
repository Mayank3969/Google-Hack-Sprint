"""Landmark preprocessing and classifier inference for the ML service."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from scripts import dataset_builder

MAX_FRAMES = dataset_builder.MAX_FRAMES
LANDMARK_SHAPE = dataset_builder.LANDMARK_SHAPE
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        Path(__file__).resolve().parents[1] / "models" / "classifier.pkl",
    )
)


class InferenceError(RuntimeError):
    """Raised when a classifier cannot produce a valid prediction."""


def _validate_landmarks(landmarks: Any) -> np.ndarray:
    """Convert and validate an incoming landmark sequence."""

    try:
        array = np.asarray(landmarks)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "landmarks must be a numeric array with shape (frames, 21, 3)"
        ) from exc
    if array.ndim != 3 or tuple(array.shape[1:]) != LANDMARK_SHAPE:
        raise ValueError(
            f"landmarks must have shape (frames, 21, 3); got {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError("landmarks must contain at least one frame")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("landmarks must contain only numeric values")
    if not np.isfinite(array).all():
        raise ValueError("landmarks must contain only finite numeric values")
    return array


def preprocess_landmarks(landmarks: Any) -> np.ndarray:
    """Return one flattened, fixed-size sample for classifier input."""

    array = _validate_landmarks(landmarks)
    fixed = dataset_builder.pad_or_truncate(array, max_len=MAX_FRAMES)
    return fixed.reshape(1, -1)


def load_classifier(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load and return the pickled classifier, with an actionable failure."""

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"classifier model does not exist: {path}")
    try:
        with path.open("rb") as artifact:
            classifier = pickle.load(artifact)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ImportError) as exc:
        raise InferenceError(f"could not load classifier model: {path}") from exc
    if not callable(getattr(classifier, "predict", None)):
        raise InferenceError("classifier model must provide a callable predict method")
    return classifier


def _prediction_label(classifier: Any, sample: np.ndarray) -> str:
    if not callable(getattr(classifier, "predict", None)):
        raise InferenceError("classifier must provide a callable predict method")
    try:
        predictions = np.asarray(classifier.predict(sample))
    except Exception as exc:  # classifier implementations expose varied exceptions
        raise InferenceError(f"classifier prediction failed: {exc}") from exc
    if predictions.ndim != 1 or len(predictions) != 1:
        raise InferenceError("classifier must return exactly one predicted label")
    label = predictions[0]
    if not isinstance(label, (str, np.str_)):
        raise InferenceError("classifier prediction must be a string label")
    return str(label)


def predict_landmarks(
    landmarks: Any,
    *,
    classifier: Any | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[str]:
    """Predict one keyword label from landmark frames.

    ``classifier`` is injectable for callers and tests. When omitted, the
    artifact at ``model_path`` is loaded lazily for this prediction.
    """

    sample = preprocess_landmarks(landmarks)
    model = classifier if classifier is not None else load_classifier(model_path)
    return [_prediction_label(model, sample)]


class InferenceService:
    """Small dependency-injectable facade for repeated predictions."""

    def __init__(self, classifier: Any | None = None, model_path: str | Path = DEFAULT_MODEL_PATH):
        self._classifier = classifier
        self._model_path = model_path

    def predict(self, landmarks: Any) -> list[str]:
        return predict_landmarks(
            landmarks, classifier=self._classifier, model_path=self._model_path
        )
