"""Load the M4 BiGRU checkpoint and classify one isolated sign window.

M5 accepts the same feature contract produced by M2/M3 and expected by M4:
``(frames, 110, 3)`` = left hand (21), right hand (21), compact face (68).
The caller may provide 1--60 mobile frames. The exact M4 temporal normalization
is applied before inference, so a short 15-FPS mobile clip follows the same
preprocessing path as training.

This module is deliberately framework/API independent. M6 can call ``predict``
from its endpoint without changing checkpoint or preprocessing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_SHAPE = (110, 3)
MAX_FRAMES = 60
PREPROCESSING_VERSION = "temporal-normalize-nonzero-v1"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "model" / "bigru" / "bigru_best.pt"


def _training_module():
    """Import the M4 module without importing PyTorch until the model is loaded."""

    return importlib.import_module("scripts.04_train_bigru")


def temporal_normalize(sample: Any, target_frames: int = MAX_FRAMES) -> np.ndarray:
    """Apply the exact M4 preprocessing to one raw mobile landmark sequence."""

    trainer = _training_module()
    if target_frames != MAX_FRAMES:
        raise ValueError("M5 target_frames must match the M4 checkpoint contract of 60")
    return trainer.temporal_normalize(sample, target_frames=target_frames)


@dataclass
class Prediction:
    label: str
    class_id: int
    confidence: float
    top_k: list[dict[str, float | int | str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "top_k": self.top_k,
        }


def _torch_load(path: Path, device: str):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch versions before the weights_only keyword
        return torch.load(path, map_location=device)


class InferenceEngine:
    """CPU/GPU-neutral checkpoint loader and predictor."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        import torch

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"M4 checkpoint does not exist: {self.checkpoint_path}; "
                "run M4 training and provide SIGNBRIDGE_MODEL_PATH"
            )
        self.device = device or os.getenv("SIGNBRIDGE_DEVICE", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SIGNBRIDGE_DEVICE=cuda but CUDA is unavailable")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")

        checkpoint = _torch_load(self.checkpoint_path, self.device)
        self._validate_checkpoint(checkpoint)
        self.checkpoint = checkpoint
        self.label_names = [str(label) for label in checkpoint["label_names"]]
        config_data = dict(checkpoint["config"])
        layers = config_data.get("layers", config_data.get("gru_layers"))
        if layers is None:
            raise ValueError("M4 checkpoint config is missing layers/gru_layers")
        trainer = _training_module()
        config = trainer.TrainConfig(
            input_size=int(config_data["input_size"]),
            hidden_size=int(config_data["hidden_size"]),
            layers=int(layers),
            dropout=float(config_data["dropout"]),
        )
        self.model = trainer.build_model(len(self.label_names), config).to(self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state"], strict=True)
        except RuntimeError as exc:
            raise ValueError("M4 checkpoint weights do not match its architecture metadata") from exc
        self.model.eval()

    @staticmethod
    def _validate_checkpoint(checkpoint: Any) -> None:
        if not isinstance(checkpoint, dict):
            raise ValueError("M4 checkpoint must contain a metadata dictionary")
        required = ("model_state", "config", "label_names", "feature_shape", "max_frames")
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise ValueError(f"M4 checkpoint is missing required fields: {', '.join(missing)}")
        if tuple(checkpoint["feature_shape"]) != FEATURE_SHAPE:
            raise ValueError("M4 checkpoint feature shape must be (110, 3)")
        if int(checkpoint["max_frames"]) != MAX_FRAMES:
            raise ValueError("M4 checkpoint max_frames must be 60")
        if checkpoint.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError(
                "M4 checkpoint preprocessing version does not match M5: "
                f"{checkpoint.get('preprocessing_version')!r}"
            )
        labels = checkpoint["label_names"]
        if not isinstance(labels, (list, tuple)) or not labels or any(not str(label) for label in labels):
            raise ValueError("M4 checkpoint label_names must be a non-empty sequence")
        config = checkpoint["config"]
        if not isinstance(config, dict) or int(config.get("input_size", -1)) != 330:
            raise ValueError("M4 checkpoint config input_size must be 330")

    def predict_details(self, landmarks: Any, top_k: int = 5) -> Prediction:
        import torch

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        arr = np.asarray(landmarks)
        if arr.ndim != 3 or tuple(arr.shape[1:]) != FEATURE_SHAPE:
            raise ValueError("landmarks must have shape (frames, 110, 3)")
        if arr.shape[0] < 1 or arr.shape[0] > MAX_FRAMES:
            raise ValueError("landmarks must contain between 1 and 60 frames")
        normalized = temporal_normalize(arr)
        batch = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            count = min(top_k, len(self.label_names))
            scores, indices = torch.topk(probabilities, count)

        top_predictions = [
            {
                "label": self.label_names[int(index)],
                "class_id": int(index),
                "confidence": float(score),
            }
            for score, index in zip(scores.cpu(), indices.cpu())
        ]
        best = top_predictions[0]
        return Prediction(
            label=str(best["label"]),
            class_id=int(best["class_id"]),
            confidence=float(best["confidence"]),
            top_k=top_predictions,
        )


@lru_cache(maxsize=4)
def get_engine(checkpoint_path: str | None = None, device: str | None = None) -> InferenceEngine:
    """Return a cached engine; use a string path so the cache is deterministic."""

    path = checkpoint_path or os.getenv("SIGNBRIDGE_MODEL_PATH") or str(DEFAULT_CHECKPOINT_PATH)
    return InferenceEngine(path, device=device)


def predict_details(landmarks: Any, top_k: int = 5) -> dict[str, Any]:
    """Predict a word and return confidence plus top-k alternatives."""

    return get_engine().predict_details(landmarks, top_k=top_k).as_dict()


def predict(landmarks: Any) -> list[str]:
    """Return the predicted isolated word in the M6-compatible list shape."""

    return [predict_details(landmarks, top_k=1)["label"]]
