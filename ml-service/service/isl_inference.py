"""B4–B6 single/batch inference for the Keras ISL classifier.

INTEGRATION_WARNING: uses real weights plus the frozen preprocessing contract
and uncalibrated thresholds. Official reevaluation/live checks must pass before
updating integration_state(). Do not turn synthetic tests into a ready flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from service.isl_model import ISLKerasEngine, get_isl_engine
from service.isl_quality import QualityPolicy
from service.isl_preprocessing import (
    PREPROCESSING_VERSION,
    PreparedSegment,
    PreprocessingError,
    SegmentQuality,
    prepare_segment,
)


STATUS_RECOGNIZED = "recognized"
STATUS_UNCERTAIN = "uncertain"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_TRACKING_LOST = "tracking_lost"
STATUS_INVALID = "invalid"


def integration_state() -> dict[str, Any]:
    """Expose honest release state after the Person A contract is wired."""
    return {
        "integration_ready": False,
        "warnings": [
            "OFFICIAL_COMPACT68_RUNTIME_REEVALUATION_PENDING",
            "QUALITY_THRESHOLDS_UNCALIBRATED",
            "REAL_VIDEO_FIXTURES_SHOW_DOMAIN_SHIFT",
            "MULTI_SIGN_CAPTURE_NOT_ACCURACY_VALIDATED",
        ],
    }


@dataclass(frozen=True)
class SegmentRequest:
    """One ordered raw buffer supplied by a browser or replay test."""

    segment_index: int
    landmarks: Any
    segment_id: str | None = None
    observed_mask: Any | None = None


@dataclass(frozen=True)
class ISLPrediction:
    segment_index: int
    status: str
    keyword: str | None
    class_id: int | None
    confidence: float | None
    top_k: list[dict[str, float | int | str]]
    quality: dict[str, int | float | bool]
    segment_id: str | None = None
    error: str | None = None
    model_version: str = "ISL_1DCNN_Transformer_v2"
    preprocessing_version: str = PREPROCESSING_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "segment_id": self.segment_id,
            "status": self.status,
            "keyword": self.keyword,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "top_k": self.top_k,
            "quality": self.quality,
            "error": self.error,
            "model_version": self.model_version,
            "preprocessing_version": self.preprocessing_version,
            **integration_state(),
        }


def _empty_quality() -> dict[str, int | float | bool]:
    return {
        "input_frames": 0,
        "valid_frames": 0,
        "hand_valid_frames": 0,
        "coordinate_coverage": 0.0,
        "left_hand_coverage": 0.0,
        "right_hand_coverage": 0.0,
        "face_coverage": 0.0,
        "has_signal": False,
    }


class ISLInferenceEngine:
    """Preprocess and classify one or more isolated-sign buffers."""

    def __init__(self, model: ISLKerasEngine | None = None, policy: QualityPolicy | None = None):
        self.model = model or get_isl_engine()
        self.policy = policy or QualityPolicy.from_env()

    @property
    def model_version(self) -> str:
        return self.model.version

    def _rejected(
        self,
        request: SegmentRequest,
        *,
        status: str,
        error: str,
        quality: SegmentQuality | None = None,
    ) -> ISLPrediction:
        return ISLPrediction(
            segment_index=request.segment_index,
            segment_id=request.segment_id,
            status=status,
            keyword=None,
            class_id=None,
            confidence=None,
            top_k=[],
            quality=(quality.as_dict() if quality else _empty_quality()),
            error=error,
            model_version=self.model_version,
        )

    def _prepare(self, request: SegmentRequest) -> PreparedSegment:
        return prepare_segment(request.landmarks, observed_mask=request.observed_mask)

    def predict_segments(
        self,
        requests: Iterable[SegmentRequest],
        *,
        top_k: int = 3,
    ) -> list[ISLPrediction]:
        """Return one result per input request, ordered by ``segment_index``.

        Invalid/low-quality segments stay in the output so a client can ask for
        the exact segment to be repeated.  Valid segments are sent to Keras in
        one batch to avoid one model call per sign.
        """

        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        values = list(requests)
        if not 1 <= len(values) <= 10:
            raise ValueError("between 1 and 10 segments are required")
        if any(type(item.segment_index) is not int or not 0 <= item.segment_index < 10 for item in values):
            raise ValueError("segment_index must be an integer from 0 to 9")
        ordered = sorted(values, key=lambda item: item.segment_index)
        if len({item.segment_index for item in ordered}) != len(ordered):
            raise ValueError("segment_index values must be unique")

        results: list[ISLPrediction | None] = [None] * len(ordered)
        prepared: list[PreparedSegment] = []
        prepared_positions: list[int] = []
        for position, request in enumerate(ordered):
            try:
                value = self._prepare(request)
            except (PreprocessingError, ValueError, TypeError) as exc:
                results[position] = self._rejected(
                    request,
                    status=STATUS_INVALID,
                    error=str(exc),
                )
                continue
            rejection = self.policy.reject_quality(value.quality)
            if rejection is None and not np.any(np.diff(value.features[:, :330], axis=0)):
                # V2 temporal centering erases a perfectly static pose. Do not
                # ask its 50-way classifier to guess from an informationless
                # tensor. A must revisit this if normalization/model changes.
                rejection = (STATUS_INSUFFICIENT_DATA, "no temporal variation remains after v2 preprocessing")
            if rejection:
                results[position] = self._rejected(
                    request,
                    status=rejection[0],
                    error=rejection[1],
                    quality=value.quality,
                )
                continue
            prepared.append(value)
            prepared_positions.append(position)

        if prepared:
            batch = np.stack([item.features for item in prepared], axis=0)
            probabilities = self.model.predict_probabilities(batch)
            for position, value, scores in zip(
                prepared_positions, prepared, probabilities, strict=True
            ):
                count = min(top_k, scores.shape[0])
                ranked = np.argsort(-scores, kind="stable")
                indices = ranked[:count]
                top_predictions = [
                    {
                        "class_id": int(index),
                        "label": self.model.labels[int(index)],
                        "confidence": float(scores[int(index)]),
                    }
                    for index in indices
                ]
                best_index = int(indices[0])
                confidence = float(scores[best_index])
                # Top-k display must NEVER change the acceptance decision.
                margin = confidence - float(scores[int(ranked[1])])
                recognized = self.policy.accepts(confidence, margin)
                request = ordered[position]
                results[position] = ISLPrediction(
                    segment_index=request.segment_index,
                    segment_id=request.segment_id,
                    status=STATUS_RECOGNIZED if recognized else STATUS_UNCERTAIN,
                    keyword=self.model.labels[best_index] if recognized else None,
                    class_id=best_index if recognized else None,
                    confidence=confidence,
                    top_k=top_predictions,
                    quality=value.quality.as_dict(),
                    error=None if recognized else "prediction confidence or margin is too low",
                    model_version=self.model_version,
                    preprocessing_version=value.preprocessing_version,
                )

        assert all(item is not None for item in results), "lost a segment result"
        return list(results)

    def predict_segment(self, request: SegmentRequest, *, top_k: int = 3) -> ISLPrediction:
        return self.predict_segments([request], top_k=top_k)[0]


def get_inference_engine() -> ISLInferenceEngine:
    """Construct the thin inference facade around the cached Keras engine."""

    return ISLInferenceEngine()
