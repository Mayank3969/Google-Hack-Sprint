"""HTTP API for landmark keyword prediction."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator

from service import inference

MAX_FRAMES = 60
POINTS_PER_FRAME = 21
COORDINATES_PER_POINT = 3


def _validate_landmark_frames(value: Any) -> list[list[list[float]]]:
    """Validate and normalize the JSON landmark shape at the HTTP boundary."""

    if not isinstance(value, list):
        raise ValueError("landmarks must be an array of frames")
    if not value:
        raise ValueError("landmarks must contain at least one frame")
    if len(value) > MAX_FRAMES:
        raise ValueError(f"landmarks cannot contain more than {MAX_FRAMES} frames")

    normalized: list[list[list[float]]] = []
    for frame_index, frame in enumerate(value):
        if not isinstance(frame, list) or len(frame) != POINTS_PER_FRAME:
            raise ValueError(
                f"frame {frame_index} must contain exactly {POINTS_PER_FRAME} points"
            )
        normalized_frame: list[list[float]] = []
        for point_index, point in enumerate(frame):
            if not isinstance(point, list) or len(point) != COORDINATES_PER_POINT:
                raise ValueError(
                    f"frame {frame_index} point {point_index} must contain exactly "
                    f"{COORDINATES_PER_POINT} coordinates"
                )
            normalized_point: list[float] = []
            for coordinate in point:
                if (
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, Real)
                    or not math.isfinite(float(coordinate))
                ):
                    raise ValueError("landmark coordinates must be finite numbers")
                normalized_point.append(float(coordinate))
            normalized_frame.append(normalized_point)
        normalized.append(normalized_frame)
    return normalized


class PredictRequest(BaseModel):
    """Payload accepted by ``POST /predict``."""

    model_config = ConfigDict(extra="forbid")

    landmarks: Any

    @field_validator("landmarks", mode="before")
    @classmethod
    def validate_landmarks(cls, value: Any) -> list[list[list[float]]]:
        return _validate_landmark_frames(value)


class PredictResponse(BaseModel):
    keywords: list[str]


Predictor = Callable[[Any], list[str]]


def get_predictor() -> Predictor:
    """Resolve inference lazily so the app can start without a model artifact."""

    return inference.predict_landmarks


app = FastAPI(title="SignBridge ML Service")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Request, _exc: RequestValidationError
) -> JSONResponse:
    """Return a serializable 422 even when rejected input contains NaN/Infinity."""

    return JSONResponse(status_code=422, content={"detail": "invalid request payload"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    predictor: Predictor = Depends(get_predictor),
) -> PredictResponse:
    try:
        keywords = predictor(request.landmarks)
    except Exception as exc:  # Keep model/provider failures as controlled API errors.
        raise HTTPException(status_code=502, detail="landmark inference failed") from exc
    if not isinstance(keywords, list) or not all(
        isinstance(item, str) for item in keywords
    ):
        raise HTTPException(
            status_code=502, detail="landmark inference returned invalid keywords"
        )
    return PredictResponse(keywords=keywords)
