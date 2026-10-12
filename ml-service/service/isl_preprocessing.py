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

import numpy as np
import warnings


PREPROCESSING_VERSION = "isl-preprocessing-v2"
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


__all__ = [
    "OUTPUT_FEATURES",
    "OUTPUT_SHAPE",
    "FACE_SOURCE_INDICES",
    "FEATURE_COLUMNS",
    "PREPROCESSING_VERSION",
    "RAW_FEATURES",
    "RAW_SHAPE",
    "TARGET_FRAMES",
    "prepare_model_input",
]
