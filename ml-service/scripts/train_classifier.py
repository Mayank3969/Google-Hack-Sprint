"""Train and persist the deterministic M4 RandomForest classifier."""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from scripts.dataset_builder import COORDINATE_FORMAT, MAX_FRAMES

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
N_ESTIMATORS = 200
CODE_VERSION = "m4-random-forest-v1"
EXPECTED_X_SHAPE = (MAX_FRAMES, 21, 3)


def _load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {dataset_path}")
    with np.load(dataset_path, allow_pickle=True) as data:
        if "X" not in data or "y" not in data:
            raise ValueError("dataset must contain X and y arrays")
        x = np.asarray(data["X"])
        y = np.asarray(data["y"])
    if x.ndim != 4 or tuple(x.shape[1:]) != EXPECTED_X_SHAPE:
        raise ValueError(
            f"X must have shape (samples, 60, 21, 3); got {x.shape}"
        )
    if len(x) == 0:
        raise ValueError("dataset must contain at least one sample")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("X must contain only finite numeric values")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError(f"y must be a 1-D array with {len(x)} labels; got {y.shape}")
    if any(str(label).strip() == "" for label in y.tolist()):
        raise ValueError("y labels must not be empty")
    return x.astype(np.float32, copy=False), y


def _validate_split(y: np.ndarray, test_size: float) -> None:
    counts = Counter(y.tolist())
    labels = len(counts)
    if labels < 2:
        raise ValueError("at least two labels are required")
    insufficient = {label: count for label, count in counts.items() if count < 2}
    if insufficient:
        raise ValueError(
            "each label needs at least 2 samples for a stratified held-out split; "
            f"insufficient counts: {insufficient}"
        )
    validation_samples = math.ceil(len(y) * test_size)
    training_samples = len(y) - validation_samples
    if validation_samples < labels or training_samples < labels:
        raise ValueError(
            f"dataset has {len(y)} samples across {labels} labels; a {test_size:.0%} "
            "held-out split cannot include every label in both sets"
        )


def train_classifier(
    dataset_path: str | Path,
    model_path: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    code_version: str = CODE_VERSION,
) -> dict[str, Any]:
    """Train a fixed RandomForest and save classifier plus artifact metadata."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    dataset = Path(dataset_path)
    output = Path(model_path)
    x, y = _load_dataset(dataset)
    _validate_split(y, test_size)
    labels = sorted({str(label) for label in y.tolist()})
    x_flat = x.reshape(len(x), -1)
    x_train, x_val, y_train, y_val = train_test_split(
        x_flat, y, test_size=test_size, stratify=y, random_state=seed
    )

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_val)
    accuracy = float(accuracy_score(y_val, predictions))
    matrix = confusion_matrix(y_val, predictions, labels=labels).tolist()
    counts = {label: int(sum(str(value) == label for value in y.tolist())) for label in labels}
    metadata: dict[str, Any] = {
        "num_samples": int(len(y)),
        "per_label_counts": counts,
        "max_frames": MAX_FRAMES,
        "coordinate_format": COORDINATE_FORMAT,
        "labels": labels,
        "seed": seed,
        "validation_accuracy": accuracy,
        "validation_samples": int(len(y_val)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as artifact:
        pickle.dump(classifier, artifact)
    metadata_path = output.with_name("classifier_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def format_report(metadata: dict[str, Any]) -> str:
    """Return the stable human-readable portion of the M4 training report."""

    lines = [
        f"samples: {metadata['num_samples']}",
        f"per-label counts: {metadata['per_label_counts']}",
        f"seed: {metadata['seed']}",
        f"validation accuracy: {metadata['validation_accuracy']:.2%}",
        f"confusion matrix (labels={metadata['confusion_matrix_labels']}):",
    ]
    lines.extend(str(row) for row in metadata["confusion_matrix"])
    return "\n".join(lines)
