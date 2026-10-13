"""Load and run the Keras INCLUDE-50 ISL classifier.

INTEGRATION_WARNING (B1/B7): real artifacts are used, but the default checkpoint
has NOT been selected by a new held-out evaluation. Person A must supply the
golden fixture, feature mapping and checkpoint selection report described in
PERSON_B_HANDOFF.md. Shape/load tests are not recognition-accuracy evidence.
"""

from __future__ import annotations

from functools import lru_cache
import json
import hashlib
import os
from pathlib import Path
from threading import Lock

import numpy as np

FEATURES_PER_FRAME = 330
MODEL_FEATURES_PER_FRAME = 660
MAX_FRAMES = 60
NUM_CLASSES = 50
MODEL_VERSION = "ISL_1DCNN_Transformer_v2"

SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = SERVICE_ROOT / "models" / "best_isl_model_v2.keras"
DEFAULT_LABEL_PATH = SERVICE_ROOT / "models" / "id_to_label.json"
DEFAULT_MANIFEST_PATH = SERVICE_ROOT / "models" / "model_manifest.json"
_LOAD_LOCK = Lock()


def _load_labels(path: Path) -> list[str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read ISL label map: {path}") from exc
    if not isinstance(raw, dict) or set(raw) != {str(i) for i in range(NUM_CLASSES)}:
        raise ValueError("ISL label map must contain exactly class IDs 0 to 49")
    labels = [raw[str(index)] for index in range(NUM_CLASSES)]
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("ISL labels must be non-empty strings")
    if len(set(labels)) != NUM_CLASSES:
        raise ValueError("ISL label map contains duplicate labels")
    return labels


class ISLKerasEngine:
    """Cached CPU/GPU-neutral wrapper for the trained Keras model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        label_path: str | Path = DEFAULT_LABEL_PATH,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    ):
        import keras
        from service.isl_layers import CUSTOM_OBJECTS

        self.model_path = Path(model_path)
        self.label_path = Path(label_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ISL Keras model does not exist: {self.model_path}")
        if not self.label_path.is_file():
            raise FileNotFoundError(f"ISL label map does not exist: {self.label_path}")

        self.labels = _load_labels(self.label_path)
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.labels_sha256 = hashlib.sha256(self.label_path.read_bytes()).hexdigest()
        from service.isl_preprocessing import PREPROCESSING_VERSION
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("artifacts", {}).get(self.model_path.name) != self.model_sha256:
            raise ValueError("model checksum is not registered in model_manifest.json")
        if manifest.get("labels_sha256") != self.labels_sha256:
            raise ValueError("label checksum differs from model_manifest.json")
        if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("model manifest preprocessing version differs from serving")
        self._lock = Lock()
        try:
            self.model = keras.models.load_model(
                self.model_path,
                custom_objects=CUSTOM_OBJECTS,
                compile=False,
            )
        except Exception as exc:
            raise RuntimeError(f"could not load ISL Keras model {self.model_path}: {exc}") from exc

        input_shape = tuple(self.model.input_shape[1:])
        output_shape = tuple(self.model.output_shape[1:])
        if input_shape != (MAX_FRAMES, MODEL_FEATURES_PER_FRAME):
            raise ValueError(
                "ISL model input must be (60, 660); "
                f"loaded {input_shape!r}"
            )
        if output_shape != (NUM_CLASSES,):
            raise ValueError(
                "ISL model output must be (50,); "
                f"loaded {output_shape!r}"
            )

    @property
    def version(self) -> str:
        return f"{MODEL_VERSION}:{self.model_sha256[:12]}"

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Run deterministic inference for a ``(N, 60, 660)`` float32 batch."""

        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 3 or tuple(arr.shape[1:]) != (MAX_FRAMES, MODEL_FEATURES_PER_FRAME):
            raise ValueError("features must have shape (batch, 60, 660)")
        if arr.shape[0] < 1:
            raise ValueError("features batch must not be empty")
        if not np.isfinite(arr).all():
            raise ValueError("features must contain only finite values")
        # Direct inference avoids a new tf.data thread pool for each HTTP batch.
        with self._lock:
            output = np.asarray(self.model(arr, training=False), dtype=np.float32)
        if output.shape != (arr.shape[0], NUM_CLASSES):
            raise RuntimeError(f"ISL model returned unexpected shape {output.shape!r}")
        if not np.isfinite(output).all():
            raise RuntimeError("ISL model returned non-finite probabilities")
        if (output < 0).any() or (output > 1).any() or not np.allclose(
            output.sum(axis=1), 1.0, atol=1e-4
        ):
            raise RuntimeError("ISL model returned invalid probability distributions")
        return output


@lru_cache(maxsize=2)
def _cached_engine(model_path: str, label_path: str, manifest_path: str) -> ISLKerasEngine:
    return ISLKerasEngine(model_path, label_path, manifest_path)


def get_isl_engine(
    model_path: str | None = None,
    label_path: str | None = None,
    manifest_path: str | None = None,
) -> ISLKerasEngine:
    """Load the selected model once per process."""

    resolved_model = model_path or os.getenv("SIGNBRIDGE_ISL_MODEL_PATH") or str(DEFAULT_MODEL_PATH)
    resolved_labels = label_path or os.getenv("SIGNBRIDGE_ISL_LABEL_PATH") or str(DEFAULT_LABEL_PATH)
    resolved_manifest = manifest_path or os.getenv("SIGNBRIDGE_ISL_MANIFEST_PATH") or str(DEFAULT_MANIFEST_PATH)
    with _LOAD_LOCK:
        return _cached_engine(resolved_model, resolved_labels, resolved_manifest)
