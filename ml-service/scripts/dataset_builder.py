"""Deterministically build the fixed-shape landmark dataset used by M4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from scripts import extract_landmarks_module

MAX_FRAMES = 60
LANDMARK_SHAPE = (21, 3)
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"})
COORDINATE_FORMAT = "MediaPipe normalized x, y, z coordinates"


def extract_landmarks(video_path: str | Path) -> np.ndarray:
    """Delegate extraction to the reusable M2 implementation."""

    return extract_landmarks_module.extract_landmarks(video_path)


def pad_or_truncate(sequence: np.ndarray, max_len: int = MAX_FRAMES) -> np.ndarray:
    """Return a float32 sequence with exactly ``max_len`` frames."""

    array = np.asarray(sequence)
    if array.ndim != 3 or tuple(array.shape[1:]) != LANDMARK_SHAPE:
        raise ValueError(
            f"landmark sequence must have shape (frames, 21, 3); got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("landmark sequence must contain only finite numeric values")
    array = array.astype(np.float32, copy=False)
    if len(array) >= max_len:
        return array[:max_len].copy()
    padding = np.zeros((max_len - len(array), *LANDMARK_SHAPE), dtype=np.float32)
    return np.concatenate((array, padding), axis=0)


def build_dataset(
    raw_dir: str | Path,
    output_path: str | Path,
    *,
    extractor: Callable[[str | Path], np.ndarray] | None = None,
) -> dict[str, object]:
    """Extract, normalize, and save videos under ``raw_dir``.

    Directories immediately below ``raw_dir`` are labels. Only files with a
    recognized video extension are considered, and both labels and filenames
    are sorted to make the output ordering reproducible.
    """

    extractor = extractor or extract_landmarks
    raw_path = Path(raw_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"raw video directory does not exist: {raw_path}")

    label_dirs = sorted((entry for entry in raw_path.iterdir() if entry.is_dir()), key=lambda p: p.name)
    if not label_dirs:
        raise ValueError(f"raw video directory has no label folders: {raw_path}")

    samples: list[np.ndarray] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    for label_dir in label_dirs:
        videos = sorted(
            (entry for entry in label_dir.iterdir() if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS),
            key=lambda p: p.name,
        )
        if not videos:
            raise ValueError(f"label folder has no video files: {label_dir.name} ({label_dir})")
        counts[label_dir.name] = 0
        for video in videos:
            try:
                sequence = extractor(video)
                fixed = pad_or_truncate(sequence)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                raise ValueError(f"invalid data in {video}: {exc}") from exc
            if len(np.asarray(sequence)) == 0:
                raise ValueError(f"invalid data in {video}: extracted sequence is empty")
            samples.append(fixed)
            labels.append(label_dir.name)
            counts[label_dir.name] += 1

    x = np.stack(samples).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=str)
    np.savez(output, X=x, y=y)
    metadata = {
        "max_frames": MAX_FRAMES,
        "coordinate_format": COORDINATE_FORMAT,
        "labels": [label_dir.name for label_dir in label_dirs],
        "per_label_counts": counts,
        "num_samples": int(len(y)),
        "shape": list(x.shape),
    }
    metadata_path = output.with_name("dataset_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
