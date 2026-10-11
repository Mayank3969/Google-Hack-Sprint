"""M6 HTTP boundary tests without requiring a trained production checkpoint."""

import importlib
import json

import numpy as np
import pytest


fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient


app_module = importlib.import_module("service.app")


class FakePrediction:
    label = "pay"
    top_k = [
        {"label": "pay", "class_id": 0, "confidence": 0.8},
        {"label": "medicine", "class_id": 1, "confidence": 0.2},
    ]


class FakeEngine:
    def predict_details(self, landmarks, top_k=1):
        assert landmarks.ndim == 3
        assert landmarks.shape[1:] == (110, 3)
        assert 1 <= landmarks.shape[0] <= 60
        assert landmarks.dtype == np.float32
        assert np.isfinite(landmarks).all()
        assert top_k == 1
        return FakePrediction()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "get_engine", lambda: FakeEngine())
    return TestClient(app_module.app)


def valid_payload(frames=2):
    return {"landmarks": np.zeros((frames, 110, 3), dtype=np.float32).tolist()}


def test_health_does_not_load_model(client, monkeypatch):
    def fail_if_loaded():
        pytest.fail("health must not initialize the inference engine")

    monkeypatch.setattr(app_module, "get_engine", fail_if_loaded)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("frames", [1, 2, 60])
def test_predict_accepts_valid_110_by_3_request(client, frames):
    response = client.post("/predict", json=valid_payload(frames=frames))
    assert response.status_code == 200
    assert response.json() == {"keywords": ["pay"]}


def test_predict_returns_one_isolated_keyword_even_with_top_k_query(client):
    response = client.post("/predict?top_k=2", json=valid_payload())
    assert response.status_code == 200
    assert response.json() == {"keywords": ["pay"]}


@pytest.mark.parametrize(
    "payload",
    [
        {"landmarks": []},
        {"landmarks": np.zeros((2, 109, 3), dtype=np.float32).tolist()},
        {"landmarks": np.zeros((61, 110, 3), dtype=np.float32).tolist()},
        {"landmarks": [[[["bad"]]]]},
        {},
    ],
)
def test_predict_rejects_invalid_payloads(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code in {400, 422}
    assert "detail" in response.json()


def test_predict_rejects_invalid_top_k(client):
    response = client.post("/predict?top_k=21", json=valid_payload())
    assert response.status_code == 422


def test_predict_rejects_oversized_request_body(client):
    body = json.dumps(valid_payload()).encode()
    body += b" " * (app_module.MAX_BODY_BYTES + 1 - len(body))
    response = client.post(
        "/predict",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "request body is too large"}


def test_predict_handles_missing_model(client, monkeypatch):
    def missing_engine():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(app_module, "get_engine", missing_engine)
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 503
    assert response.json() == {"detail": "trained model is not installed"}


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
@pytest.mark.parametrize("stage", ["load", "inference"])
def test_predict_handles_unavailable_model(client, monkeypatch, error_type, stage):
    class UnavailableEngine:
        def predict_details(self, landmarks, top_k=1):
            raise error_type("private model failure")

    def unavailable_engine():
        if stage == "load":
            raise error_type("private checkpoint failure")
        return UnavailableEngine()

    monkeypatch.setattr(app_module, "get_engine", unavailable_engine)
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 503
    assert response.json() == {"detail": "model is unavailable"}
