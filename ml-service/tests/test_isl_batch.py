"""B5/B8 HTTP tests and a separate REAL-ARTIFACT HTTP smoke check.

INTEGRATION_WARNING: doubles exercise boundaries; real weights on generated
input prove execution only. Neither category establishes recognition accuracy.
Person A supplies the independent release fixture described in the handoff.
"""

import asyncio

from fastapi.testclient import TestClient
import numpy as np
import pytest

from service import app as app_module
from service.body_limit import BodyLimitMiddleware
from service.isl_inference import ISLInferenceEngine
from test_isl_inference import ModelDouble


def segment(index=0, frames=8):
    return {"segment_index": index, "segment_id": f"s{index}",
            "landmarks": np.random.default_rng(index).random((frames, 110, 3)).tolist()}


@pytest.fixture
def client(monkeypatch):
    engine = ISLInferenceEngine(ModelDouble())
    monkeypatch.setattr(app_module, "get_inference_engine", lambda: engine)
    return TestClient(app_module.app)


def test_complete_ordered_batch_and_integration_warning(client):
    response = client.post("/v2/predict/batch", json={"capture_id": "c1", "segments": [segment(1), segment(0)]})
    assert response.status_code == 200
    body = response.json()
    assert body["complete"]
    assert body["keywords"] == ["sign0", "sign0"]  # repeated signs are intentional
    assert [r["segment_id"] for r in body["results"]] == ["s0", "s1"]
    assert not body["integration_ready"]
    assert "OFFICIAL_COMPACT68_RUNTIME_REEVALUATION_PENDING" in body["warnings"]


def test_mixed_invalid_buffers_never_expose_partial_keywords(client):
    bad = {"segment_index": 1, "landmarks": []}
    response = client.post("/v2/predict/batch", json={"capture_id": "c1", "segments": [segment(), bad]})
    body = response.json()
    assert response.status_code == 200
    assert not body["complete"]
    assert body["keywords"] == []
    assert body["results"][1]["status"] == "invalid"


def test_float32_overflow_is_an_invalid_slot_not_a_service_failure(client):
    bad = segment()
    bad["landmarks"][0][0][0] = 10**400
    response = client.post("/v2/predict/batch", json={"capture_id": "c1", "segments": [bad]})
    assert response.status_code == 200
    body = response.json()
    assert not body["complete"]
    assert body["keywords"] == []
    assert body["results"][0]["status"] == "invalid"


def test_batch_and_index_bounds(client):
    for segments in [[], [segment(0), segment(0)], [segment(i) for i in range(11)],
                     [{**segment(), "segment_index": True}], [segment(0, 451)]]:
        response = client.post("/v2/predict/batch", json={"capture_id": "c", "segments": segments})
        assert response.status_code == 422
    response = client.post("/v2/predict/batch", json={"capture_id": "c", "segments": [segment(i) for i in range(10)]})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 10


def test_batch_frame_budget_is_aggregate_and_inclusive(client):
    response = client.post(
        "/v2/predict/batch",
        json={"capture_id": "c", "segments": [segment(0, 225), segment(1, 225)]},
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2

    response = client.post(
        "/v2/predict/batch",
        json={"capture_id": "c", "segments": [segment(0, 225), segment(1, 226)]},
    )
    assert response.status_code == 422


def test_duplicate_ids_and_blank_capture_rejected(client):
    first, second = segment(0), segment(1)
    second["segment_id"] = first["segment_id"]
    assert client.post("/v2/predict/batch", json={"capture_id": "c", "segments": [first, second]}).status_code == 422
    assert client.post("/v2/predict/batch", json={"capture_id": " ", "segments": [first]}).status_code == 422


def test_null_mask_and_single_endpoint(client):
    value = segment()
    value["landmarks"][0][0][0] = None
    assert client.post("/v2/predict", json=value).status_code == 200
    value["observed_mask"] = np.ones((8, 110, 3), dtype=int).tolist()
    assert client.post("/v2/predict", json=value).json()["status"] == "invalid"
    assert client.post("/v2/predict?top_k=11", json=value).status_code == 422


def test_readiness_is_not_integration_readiness(client):
    value = client.get("/v2/readiness").json()
    assert value["model_ready"] and not value["integration_ready"]


def test_runtime_errors_are_controlled(client, monkeypatch):
    def broken():
        raise RuntimeError("sensitive internal path")
    monkeypatch.setattr(app_module, "get_inference_engine", broken)
    response = client.post("/v2/predict", json=segment())
    assert response.status_code == 503
    assert "sensitive" not in response.text


def test_single_long_raw_buffer_is_resampled_not_truncated(client):
    response = client.post("/v2/predict", json=segment(frames=154))
    assert response.status_code == 200
    assert response.json()["quality"]["input_frames"] == 154


def test_chunked_body_limit_before_downstream():
    called = []
    async def downstream(scope, receive, send):
        called.append(True)
    messages = iter([
        {"type": "http.request", "body": b"a" * 7, "more_body": True},
        {"type": "http.request", "body": b"b" * 7, "more_body": False},
    ])
    async def receive():
        return next(messages)
    sent = []
    async def send(message):
        sent.append(message)
    middleware = BodyLimitMiddleware(downstream, default_limit=10, batch_limit=10)
    asyncio.run(middleware({"type": "http", "path": "/v2/predict/batch", "headers": []}, receive, send))
    assert sent[0]["status"] == 413
    assert not called


def test_content_length_limits(client):
    assert client.post("/v2/predict", content=b"{}", headers={"content-length": "-1"}).status_code == 400
    assert client.post("/v2/predict/batch", content=b"x" * (app_module.MAX_BATCH_BODY_BYTES + 1)).status_code == 413


@pytest.mark.real_model
def test_real_model_through_http_and_empty_rejection():
    # No monkeypatch: exercise the real Keras loader, preprocessing and HTTP.
    with TestClient(app_module.app) as client:
        value = segment()
        response = client.post("/v2/predict", json=value)
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in {"recognized", "uncertain"}
        assert len(result["top_k"]) == 3
        assert not result["integration_ready"]
        value["landmarks"] = np.zeros((8, 110, 3)).tolist()
        response = client.post("/v2/predict", json=value)
        assert response.json()["status"] == "insufficient_data"
