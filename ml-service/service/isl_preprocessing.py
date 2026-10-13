"""Versioned preprocessing for the compact-68 ISL model.

The implementation intentionally mirrors ``Train_ISL_1DCNN_Transformer_compact68_v2``:
raw frames are resampled with integer ``linspace`` indices, centered per feature
after resampling, and augmented with first-order velocity.  The input feature
order is preserved: hand slot 0, hand slot 1, then the 68 compact face points;
each point is represented as ``x, y, z``.

Version: ``isl-preprocessing-v2``.
Tolerance: values must be real numeric values and may be NaN (a missing
landmark).  NaNs are handled exactly as in the notebook: finite values are
centered using ``nanmean`` and missing values become zero.  Infinities are
rejected.  Values must also remain finite after conversion to float32.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Any

import numpy as np
import warnings


PREPROCESSING_VERSION = "isl-preprocessing-v2"
MAX_RAW_FRAMES = 450
TARGET_FRAMES = 60
POINTS_PER_FRAME = 110
COORDINATES_PER_POINT = 3
RAW_FEATURES = POINTS_PER_FRAME * COORDINATES_PER_POINT
OUTPUT_FEATURES = RAW_FEATURES * 2
RAW_SHAPE = (POINTS_PER_FRAME, COORDINATES_PER_POINT)
OUTPUT_SHAPE = (TARGET_FRAMES, OUTPUT_FEATURES)
FACE_SOURCE_INDICES = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    1, 2, 4, 5, 6, 168,
)
FEATURE_COLUMNS = tuple(
    f"hand_{hand}_{point}_{axis}"
    for hand in (0, 1)
    for point in range(21)
    for axis in "xyz"
) + tuple(
    f"face_{source_index}_{axis}"
    for source_index in FACE_SOURCE_INDICES
    for axis in "xyz"
)
assert len(FACE_SOURCE_INDICES) == 68
assert len(FEATURE_COLUMNS) == RAW_FEATURES


class PreprocessingError(ValueError):
    """A raw segment cannot satisfy the frozen preprocessing contract."""


@dataclass(frozen=True)
class SegmentQuality:
    """Observation quality calculated before missing values are transformed."""

    input_frames: int
    valid_frames: int
    hand_valid_frames: int
    coordinate_coverage: float
    left_hand_coverage: float
    right_hand_coverage: float
    face_coverage: float
    has_signal: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "input_frames": self.input_frames,
            "valid_frames": self.valid_frames,
            "hand_valid_frames": self.hand_valid_frames,
            "coordinate_coverage": self.coordinate_coverage,
            "left_hand_coverage": self.left_hand_coverage,
            "right_hand_coverage": self.right_hand_coverage,
            "face_coverage": self.face_coverage,
            "has_signal": self.has_signal,
        }


@dataclass(frozen=True)
class PreparedSegment:
    features: np.ndarray
    quality: SegmentQuality
    preprocessing_version: str = PREPROCESSING_VERSION


def _as_float32(raw_sequence: object) -> np.ndarray:
    """Validate a raw sequence and return a float32 ``(T, 330)`` view/copy."""
    try:
        values = np.asarray(raw_sequence)
    except (TypeError, ValueError):
        raise ValueError("raw_sequence must be a rectangular numeric array") from None

    if values.ndim == 3:
        if values.shape[1:] != RAW_SHAPE:
            raise ValueError(f"raw_sequence must have shape (T, 110, 3) or (T, 330); got {values.shape}")
        values = values.reshape(values.shape[0], RAW_FEATURES)
    elif values.ndim == 2:
        if values.shape[1] != RAW_FEATURES:
            raise ValueError(f"raw_sequence must have shape (T, 110, 3) or (T, 330); got {values.shape}")
    else:
        raise ValueError(f"raw_sequence must have shape (T, 110, 3) or (T, 330); got {values.shape}")

    if values.shape[0] < 1:
        raise ValueError("raw_sequence must contain at least one frame")
    # Object arrays containing numeric strings are deliberately not accepted:
    # the CSV audit performs its own explicit text-to-number validation.
    if values.dtype.kind not in "fiu":
        raise ValueError("raw_sequence must contain real numeric values")
    if np.isinf(values).any():
        raise ValueError("raw_sequence must not contain infinite values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(np.float32, copy=False)
    if np.isinf(result).any():
        raise ValueError("raw_sequence values must fit finite float32 values")
    return result


def _normalize_like_notebook(sequence: np.ndarray) -> np.ndarray:
    """Apply the notebook's post-resampling mean-centering and NaN policy."""
    mask = ~np.isnan(sequence)
    if np.any(mask):
        # An all-missing feature has no mean; the notebook subsequently maps
        # its masked values to zero.  Suppress only nanmean's expected warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_value = np.nanmean(sequence, axis=0, keepdims=True)
        # np.where deliberately maps missing values to zero.  For an entirely
        # missing feature nanmean is NaN, but the false branch still yields 0.
        sequence = np.where(mask, sequence - mean_value, 0.0)
    else:
        sequence = np.nan_to_num(sequence, nan=0.0)
    return sequence.astype(np.float32, copy=False)


def prepare_model_input(raw_sequence: object) -> np.ndarray:
    """Prepare one raw landmark sequence for the v2 model.

    Parameters
    ----------
    raw_sequence:
        A numeric ``(T, 110, 3)`` array or its flattened ``(T, 330)`` form.
        NaNs represent missing coordinates and are accepted; infinities and
        non-numeric/object arrays are rejected.

    Returns
    -------
    numpy.ndarray
        Float32 array of exactly ``(60, 660)`` containing
        ``[position, velocity]``.
    """
    flat = _as_float32(raw_sequence)
    n_frames = flat.shape[0]
    # Keep this expression byte-for-byte equivalent to the notebook.  Integer
    # truncation is intentional and duplicate indices are expected for short
    # sequences.
    indices = np.linspace(0, n_frames - 1, TARGET_FRAMES).astype(int)
    seq_resampled = flat[indices]
    seq_norm = _normalize_like_notebook(seq_resampled)
    velocity = np.diff(seq_norm, axis=0, prepend=seq_norm[:1, ...])
    output = np.concatenate([seq_norm, velocity], axis=-1)
    if output.shape != OUTPUT_SHAPE or output.dtype != np.float32:
        raise RuntimeError("internal preprocessing contract violation")
    return output


def _raw_3d(raw_sequence: Any, observed_mask: Any | None) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the serving envelope without changing Person A's transform."""
    def invalid_scalar(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)) or isinstance(value, str):
            return True
        if isinstance(value, np.ndarray) and value.dtype.kind == "O":
            return any(invalid_scalar(item) for item in value.flat)
        if isinstance(value, (list, tuple)):
            return any(invalid_scalar(item) for item in value)
        return False

    if invalid_scalar(raw_sequence):
        raise PreprocessingError("raw_sequence must contain real numeric values or null")
    try:
        values = np.asarray(raw_sequence)
    except (TypeError, ValueError) as exc:
        raise PreprocessingError("raw_sequence must be a rectangular numeric array") from exc
    if values.ndim != 3 or tuple(values.shape[1:]) != RAW_SHAPE:
        raise PreprocessingError("raw_sequence must have shape (T, 110, 3)")
    if not 1 <= values.shape[0] <= MAX_RAW_FRAMES:
        raise PreprocessingError(f"raw_sequence must contain 1–{MAX_RAW_FRAMES} frames")
    if values.dtype.kind == "O":
        normalized = np.empty(values.shape, dtype=np.float32)
        for index, item in enumerate(values.flat):
            if item is None:
                normalized.flat[index] = np.nan
            elif isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real):
                raise PreprocessingError("raw_sequence must contain real numeric values or null")
            else:
                try:
                    normalized.flat[index] = item
                except (OverflowError, TypeError, ValueError):
                    raise PreprocessingError(
                        "raw_sequence values must fit finite float32 values"
                    ) from None
        values = normalized
    elif values.dtype.kind not in "fiu":
        raise PreprocessingError("raw_sequence must contain real numeric values")
    if np.isinf(values).any():
        raise PreprocessingError("raw_sequence must not contain infinite values")
    with np.errstate(over="ignore", invalid="ignore"):
        raw = values.astype(np.float32, copy=True)
    # JSON numbers may be finite at Python/float64 precision but overflow to
    # infinity when the serving tensor is converted to float32.  Do not turn
    # that precision loss into detector missingness: the HTTP contract accepts
    # finite coordinates (or null), and an out-of-range coordinate is invalid.
    if np.isinf(raw).any():
        raise PreprocessingError("raw_sequence values must fit finite float32 values")
    observed = np.isfinite(raw)
    if observed_mask is not None:
        mask = np.asarray(observed_mask)
        if mask.shape != raw.shape or mask.dtype.kind != "b":
            raise PreprocessingError("observed_mask must be boolean with shape (T, 110, 3)")
        observed &= mask
    else:
        # The extractor's missing modality convention is an all-zero block.
        # Individual zero coordinates remain valid observations.
        for start, end in ((0, 21), (21, 42), (42, 110)):
            absent = np.all(raw[:, start:end] == 0, axis=(1, 2))
            observed[absent, start:end] = False
    return raw, observed


def _quality(raw: np.ndarray, observed: np.ndarray) -> SegmentQuality:
    usable_points = observed.all(axis=-1)
    frame_valid = usable_points.any(axis=1)
    hand_valid = usable_points[:, :42].any(axis=1)

    def block(start: int, end: int) -> float:
        values = observed[:, start:end]
        return float(values.mean()) if values.size else 0.0

    return SegmentQuality(
        input_frames=int(raw.shape[0]),
        valid_frames=int(frame_valid.sum()),
        hand_valid_frames=int(hand_valid.sum()),
        coordinate_coverage=float(observed.mean()),
        left_hand_coverage=block(0, 21),
        right_hand_coverage=block(21, 42),
        face_coverage=block(42, 110),
        has_signal=bool(usable_points.any()),
    )


def prepare_segment(raw_sequence: Any, *, observed_mask: Any | None = None) -> PreparedSegment:
    """Return frozen v2 features plus quality for the serving layer.

    Person A's ``prepare_model_input`` remains the sole source of feature math.
    This adapter only maps detector-zero/mask absence to NaN for quality-safe
    serving, then delegates the exact frozen transform.
    """
    raw, observed = _raw_3d(raw_sequence, observed_mask)
    quality = _quality(raw, observed)
    masked = raw.reshape(raw.shape[0], RAW_FEATURES)
    masked_observed = observed.reshape(observed.shape[0], RAW_FEATURES)
    masked[~masked_observed] = np.nan
    features = prepare_model_input(masked)
    return PreparedSegment(features=features, quality=quality)


__all__ = [
    "OUTPUT_FEATURES",
    "OUTPUT_SHAPE",
    "FACE_SOURCE_INDICES",
    "FEATURE_COLUMNS",
    "PREPROCESSING_VERSION",
    "PreprocessingError",
    "PreparedSegment",
    "RAW_FEATURES",
    "RAW_SHAPE",
    "SegmentQuality",
    "TARGET_FRAMES",
    "MAX_RAW_FRAMES",
    "prepare_segment",
    "prepare_model_input",
]
