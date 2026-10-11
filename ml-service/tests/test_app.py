from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from service import app as app_module


def payload(frames: int = 1) -> dict:
    return {
        "landmarks": [
            [[0.1, 0.2, -0.01] for _ in range(21)] for _ in range(frames)
        ]
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.inference, "predict_landmarks", lambda landmarks: ["wave"])
    return TestClient(app_module.app)


def test_predict_valid_payload(client):
    response = client.post("/predict", json=payload())

    assert response.status_code == 200
    assert response.json() == {"keywords": ["wave"]}


@pytest.mark.parametrize(
    "body",
    [
        {"landmarks": []},
        {"landmarks": [[[0.0, 0.0, 0.0] for _ in range(20)]]},
        {"landmarks": [[[0.0, 0.0] for _ in range(21)]]},
        {"landmarks": [[0.0] * 21]},
        {"landmarks": [[["not-a-number", 0.0, 0.0] for _ in range(21)]]},
        {"landmarks": [[[math.nan, 0.0, 0.0] for _ in range(21)]]},
        {},
    ],
)
def test_predict_rejects_invalid_payload(client, body):
    has_nonfinite_value = any(
        isinstance(coordinate, float) and not math.isfinite(coordinate)
        for frame in body.get("landmarks", [])
        if isinstance(frame, list)
        for point in frame
        if isinstance(point, list)
        for coordinate in point
    )
    if has_nonfinite_value:
        response = client.post(
            "/predict",
            content=json.dumps(body, allow_nan=True),
            headers={"content-type": "application/json"},
        )
    else:
        response = client.post("/predict", json=body)
    assert response.status_code == 422


def test_predict_rejects_more_than_60_frames(client):
    response = client.post("/predict", json=payload(61))

    assert response.status_code == 422


def test_predict_maps_missing_model_to_bad_gateway(monkeypatch):
    def missing_model(_landmarks):
        raise FileNotFoundError("classifier model does not exist")

    monkeypatch.setattr(app_module.inference, "predict_landmarks", missing_model)
    response = TestClient(app_module.app).post("/predict", json=payload())

    assert response.status_code == 502
    assert response.json() == {"detail": "landmark inference failed"}


def test_predict_maps_inference_failure_to_bad_gateway(monkeypatch):
    def failed_inference(_landmarks):
        raise RuntimeError("classifier failed")

    monkeypatch.setattr(app_module.inference, "predict_landmarks", failed_inference)
    response = TestClient(app_module.app).post("/predict", json=payload())

    assert response.status_code == 502
