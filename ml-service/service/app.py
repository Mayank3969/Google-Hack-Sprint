"""FastAPI HTTP boundary for the SignBridge M5 inference wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from service.inference import FEATURE_SHAPE, MAX_FRAMES, get_engine


MAX_BODY_BYTES = 512 * 1024
MAX_TOP_K = 20

app = FastAPI(
    title="SignBridge ML Service",
    version="1.0.0",
    description="Predict an isolated sign word from MediaPipe landmark sequences.",
)


class LandmarkInput(BaseModel):
    """JSON request envelope; nested landmark validation happens explicitly."""

    landmarks: Any


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


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Reject oversized requests before FastAPI parses the JSON body.

    Content-Length is only an early check: clients may use chunked transfer
    encoding, so the receive callable is also capped while the request body is
    consumed by Starlette/FastAPI.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            content_length_value = int(content_length)
            if content_length_value < 0:
                return await _json_error(400, "invalid content-length header")
            if content_length_value > MAX_BODY_BYTES:
                return await _json_error(413, "request body is too large")
        except ValueError:
            return await _json_error(400, "invalid content-length header")

    received = 0
    receive = request._receive

    async def limited_receive():
        nonlocal received
        message = await receive()
        if message.get("type") == "http.request":
            received += len(message.get("body", b""))
            if received > MAX_BODY_BYTES:
                raise _RequestBodyTooLarge
        return message

    request._receive = limited_receive
    try:
        return await call_next(request)
    except _RequestBodyTooLarge:
        return await _json_error(413, "request body is too large")


class _RequestBodyTooLarge(Exception):
    """Internal control-flow exception used by the body-size middleware."""


async def _json_error(status_code: int, detail: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})


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
