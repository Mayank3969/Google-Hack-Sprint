"""Legacy /predict plus versioned Keras ISL endpoints.

INTEGRATION_WARNING (B8): /v2 results remain evidence-pending until official
reevaluation/calibration/live checks pass. Clients must inspect complete AND
integration_ready before speech.
Do not replace the legacy path implicitly; migrate callers during integration.
"""

from __future__ import annotations

from typing import Any
import os

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StrictStr, StrictInt

from service.inference import FEATURE_SHAPE, MAX_FRAMES, get_engine
from service.isl_inference import (
    ISLInferenceEngine,
    SegmentRequest,
    get_inference_engine,
    integration_state,
)
from service.body_limit import BodyLimitMiddleware


MAX_BODY_BYTES = 512 * 1024
MAX_BATCH_BODY_BYTES = 8 * 1024 * 1024
MAX_TOP_K = 20
MAX_ISL_SEGMENTS = 10

app = FastAPI(
    title="SignBridge ML Service",
    version="1.0.0",
    description="Predict an isolated sign word from MediaPipe landmark sequences.",
)

app.add_middleware(BodyLimitMiddleware, default_limit=MAX_BODY_BYTES, batch_limit=MAX_BATCH_BODY_BYTES)
allowed_origins = [v.strip() for v in os.getenv("SIGNBRIDGE_ALLOWED_ORIGINS", "").split(",") if v.strip()]
if allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class LandmarkInput(BaseModel):
    """JSON request envelope; nested landmark validation happens explicitly."""

    landmarks: Any


class ISLSegmentInput(BaseModel):
    """One raw landmark buffer for the Keras INCLUDE-50 model."""

    segment_index: StrictInt = Field(ge=0, le=MAX_ISL_SEGMENTS - 1)
    segment_id: StrictStr | None = Field(default=None, max_length=128)
    landmarks: Any
    observed_mask: Any | None = None


class ISLBatchInput(BaseModel):
    """Ordered isolated-sign buffers from one recording session."""

    capture_id: StrictStr = Field(min_length=1, max_length=128)
    segments: list[ISLSegmentInput] = Field(
        min_length=1,
        max_length=MAX_ISL_SEGMENTS,
    )


def validate_landmarks(value: Any) -> np.ndarray:
    """Validate the mobile payload and return a finite float32 array."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="landmarks must be a rectangular numeric array") from None
    if array.ndim != 3 or tuple(array.shape[1:]) != FEATURE_SHAPE:
        raise HTTPException(
            status_code=422,
            detail=f"landmarks must have shape (frames, {FEATURE_SHAPE[0]}, {FEATURE_SHAPE[1]})",
        )
    if array.shape[0] < 1 or array.shape[0] > MAX_FRAMES:
        raise HTTPException(status_code=422, detail="landmarks must contain between 1 and 60 frames")
    if array.dtype.kind not in "fiu":
        raise HTTPException(status_code=422, detail="landmark coordinates must be numeric")
    try:
        array = array.astype(np.float32, copy=False)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(status_code=422, detail="landmark coordinates must be finite float values") from None
    if not np.isfinite(array).all():
        raise HTTPException(status_code=422, detail="landmark coordinates must be finite")
    return array


@app.get("/health")
def health() -> dict[str, str]:
    """Cheap liveness check; it does not load the checkpoint or run inference."""

    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(payload: LandmarkInput, top_k: int = 1) -> dict[str, list[str]]:
    """Predict an isolated word from 1–60 frames of 110×3 landmarks."""

    # Keep accepting the old optional query bound, but never expose M5's
    # debugging alternatives as user-facing keyword output.
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(status_code=422, detail=f"top_k must be between 1 and {MAX_TOP_K}")
    landmarks = validate_landmarks(payload.landmarks)
    try:
        details = get_engine().predict_details(landmarks, top_k=1)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="trained model is not installed") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="model is unavailable") from exc
    return {"keywords": [details.label]}


def _isl_engine_or_503() -> ISLInferenceEngine:
    try:
        return get_inference_engine()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="ISL Keras model is not installed") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="ISL Keras model is unavailable") from exc


@app.post("/v2/predict")
def isl_predict_endpoint(payload: ISLSegmentInput, top_k: int = Query(3, ge=1, le=10)) -> dict[str, Any]:
    """Predict one raw landmark buffer using the Keras ISL model."""

    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 10")
    engine = _isl_engine_or_503()
    try:
        result = engine.predict_segment(
            SegmentRequest(
                segment_index=payload.segment_index,
                segment_id=payload.segment_id,
                landmarks=payload.landmarks,
                observed_mask=payload.observed_mask,
            ),
            top_k=top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="ISL prediction failed") from exc
    return result.as_dict()


@app.post("/v2/predict/batch")
def isl_batch_predict_endpoint(payload: ISLBatchInput, top_k: int = Query(3, ge=1, le=10)) -> dict[str, Any]:
    """Predict all isolated-sign buffers from one recording session."""

    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 10")
    indices = [segment.segment_index for segment in payload.segments]
    if len(set(indices)) != len(indices):
        raise HTTPException(status_code=422, detail="segment_index values must be unique")
    identifiers = [s.segment_id for s in payload.segments if s.segment_id is not None]
    if len(set(identifiers)) != len(identifiers):
        raise HTTPException(status_code=422, detail="segment_id values must be unique when supplied")
    if not payload.capture_id.strip():
        raise HTTPException(status_code=422, detail="capture_id must not be blank")
    total_frames = sum(len(s.landmarks) for s in payload.segments if isinstance(s.landmarks, list))
    if total_frames > 450:
        raise HTTPException(status_code=422, detail="batch exceeds 450 total raw frames")
    engine = _isl_engine_or_503()
    requests = [
        SegmentRequest(
            segment_index=segment.segment_index,
            segment_id=segment.segment_id,
            landmarks=segment.landmarks,
            observed_mask=segment.observed_mask,
        )
        for segment in payload.segments
    ]
    try:
        results = engine.predict_segments(requests, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="ISL batch prediction failed") from exc
    results_json = [result.as_dict() for result in results]
    complete = all(result["status"] == "recognized" for result in results_json)
    keywords = [
        result.keyword
        for result in results
        if result.status == "recognized" and result.keyword is not None
    ]
    return {
        "capture_id": payload.capture_id,
        "complete": complete,
        "keywords": keywords if complete else [],
        "results": results_json,
        **integration_state(),
    }


@app.get("/v2/readiness")
def isl_readiness() -> dict[str, Any]:
    """Model-load readiness is deliberately separate from integration readiness."""
    engine = _isl_engine_or_503()
    return {"model_ready": True, "model_version": engine.model_version, **integration_state()}
