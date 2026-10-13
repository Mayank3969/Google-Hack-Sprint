"""Root-level cross-service checks for the agreed capture-to-conversation flow.

These tests intentionally use FastAPI ``TestClient`` and deterministic model,
Gemini, TTS, STT, and Firestore paths.  They verify the contracts between
services without claiming recognition quality or calling a cloud provider.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import sys
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ml-service"))

from backend.main import app as backend_app  # noqa: E402
from backend.routes import process as process_route  # noqa: E402
from backend.services import firestore_client  # noqa: E402
from service import app as ml_app  # noqa: E402
from service.isl_inference import ISLInferenceEngine  # noqa: E402


class OrderedModelDouble:
    """Return one known label per valid row, in the model batch's row order."""

    version = "ROOT-TEST-ONLY"
    labels = ["pay", "medicine", "?"] + [f"unused{i}" for i in range(47)]

    def __init__(self, *, confidence: float = 0.9, runner_up: float = 0.08):
        self.confidence = confidence
        self.runner_up = runner_up
        self.calls: list[np.ndarray] = []

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        self.calls.append(features.copy())
        remaining = 1.0 - self.confidence - self.runner_up
        probabilities = np.full(
            (len(features), len(self.labels)),
            remaining / (len(self.labels) - 2),
            dtype=np.float32,
        )
        for row in range(len(features)):
            primary = row % len(self.labels)
            secondary = (primary + 3) % len(self.labels)
            probabilities[row, primary] = self.confidence
            probabilities[row, secondary] = self.runner_up
        return probabilities


def _landmarks(seed: int) -> list[list[list[float]]]:
    """Create non-static, finite raw landmarks accepted by the v2 preprocessor."""

    return np.random.default_rng(seed).random((12, 110, 3), dtype=np.float32).tolist()


def _wav_base64(*, silent: bool = False) -> str:
    stream = BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        sample = b"\x00\x00" if silent else b"\x01\x00"
        wav.writeframes(sample * 160)
    return base64.b64encode(stream.getvalue()).decode("ascii")


@pytest.fixture
def service_clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Provide isolated service clients with all external boundaries mocked."""

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "0")
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("TTS_EN_LANGUAGE_CODE", "en-IN")
    monkeypatch.setenv("TTS_HI_LANGUAGE_CODE", "hi-IN")
    monkeypatch.setenv("STT_LANGUAGE_CODE", "en-IN")
    monkeypatch.setenv("TTS_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("STT_TIMEOUT_SECONDS", "10")
    monkeypatch.setattr(process_route, "AUDIO_DIR", tmp_path)
    firestore_client.reset_mock_repository()
    with TestClient(ml_app.app) as ml_client, TestClient(backend_app) as backend_client:
        yield ml_client, backend_client, tmp_path


def _batch_payload() -> dict[str, object]:
    # Deliberately submit out of order: the response contract sorts by index.
    return {
        "capture_id": "capture-root-flow",
        "segments": [
            {"segment_index": 2, "segment_id": "seg-2", "landmarks": _landmarks(2)},
            {"segment_index": 0, "segment_id": "seg-0", "landmarks": _landmarks(0)},
            {"segment_index": 1, "segment_id": "seg-1", "landmarks": _landmarks(1)},
        ],
    }


def _backend_capture_payload(batch_body: dict[str, object]) -> dict[str, object]:
    """Adapt the ML quality field to the backend's deliberately smaller DTO."""

    results = []
    for item in batch_body["results"]:
        quality = item["quality"]
        results.append(
            {
                "segment_index": item["segment_index"],
                "status": item["status"],
                "keyword": item["keyword"],
                "confidence": item["confidence"],
                "coverage": quality["coordinate_coverage"],
                "top_k": [
                    {"label": candidate["label"], "confidence": candidate["confidence"]}
                    for candidate in item["top_k"]
                ],
            }
        )
    return {"capture_id": batch_body["capture_id"], "results": results}


def test_agreed_batch_to_process_transcribe_and_session_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    service_clients,
) -> None:
    """A complete batch is ordered, localized, spoken, transcribed, and stored."""

    model = OrderedModelDouble()
    monkeypatch.setattr(ml_app, "get_inference_engine", lambda: ISLInferenceEngine(model))
    ml_client, backend_client, audio_dir = service_clients

    assert ml_client.get("/health").json() == {"status": "ok"}
    assert backend_client.get("/health").json() == {"status": "ok"}

    batch = ml_client.post("/v2/predict/batch", json=_batch_payload())
    assert batch.status_code == 200, batch.text
    batch_body = batch.json()
    assert batch_body["capture_id"] == "capture-root-flow"
    assert batch_body["complete"] is True
    assert batch_body["keywords"] == ["pay", "medicine", "?"]
    assert [item["segment_index"] for item in batch_body["results"]] == [0, 1, 2]
    assert [item["segment_id"] for item in batch_body["results"]] == [
        "seg-0", "seg-1", "seg-2"
    ]
    assert [item["status"] for item in batch_body["results"]] == [
        "recognized", "recognized", "recognized"
    ]
    assert len(model.calls) == 1
    assert model.calls[0].shape == (3, 60, 660)
    assert batch_body["integration_ready"] is False

    processed = backend_client.post(
        "/process", json=_backend_capture_payload(batch_body)
    )
    assert processed.status_code == 200, processed.text
    process_body = processed.json()
    assert process_body["sentence"] == "Can I pay for this medicine?"
    assert process_body["hindi_sentence"]
    assert process_body["subtitles"] == [
        {"language": "en-IN", "text": "Can I pay for this medicine?"},
        {"language": "hi-IN", "text": "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?"},
    ]
    assert process_body["playback_order"] == ["en-IN", "hi-IN"]
    assert [item["language"] for item in process_body["audio_segments"]] == [
        "en-IN", "hi-IN"
    ]
    assert process_body["audio_url"] == process_body["audio_segments"][0]["audio_url"]
    assert len(list(audio_dir.glob("*.mp3"))) == 2

    room_response = backend_client.post("/session/create")
    assert room_response.status_code == 200
    room_id = room_response.json()["room_id"]
    assert backend_client.post(
        f"/session/{room_id}/message",
        json={"sender": "signer", "text": process_body["sentence"]},
    ).status_code == 200

    transcribed = backend_client.post("/transcribe", json={"audio": _wav_base64()})
    assert transcribed.status_code == 200
    assert transcribed.json()["text"] == "[MOCK] Yes, that is fine."
    assert backend_client.post(
        f"/session/{room_id}/message",
        json={"sender": "speaker", "text": transcribed.json()["text"]},
    ).status_code == 200

    messages = backend_client.get(f"/session/{room_id}/messages")
    assert messages.status_code == 200
    assert [(item["sender"], item["text"]) for item in messages.json()["messages"]] == [
        ("signer", "Can I pay for this medicine?"),
        ("speaker", "[MOCK] Yes, that is fine."),
    ]
    assert model.calls[0].shape[0] == 3  # backend never orchestrates another ML call


def test_uncertain_batch_is_incomplete_and_cannot_cross_into_process(
    monkeypatch: pytest.MonkeyPatch,
    service_clients,
) -> None:
    """Abstention keeps keywords empty and blocks the sentence boundary."""

    model = OrderedModelDouble(confidence=0.51, runner_up=0.49)
    monkeypatch.setattr(ml_app, "get_inference_engine", lambda: ISLInferenceEngine(model))
    ml_client, backend_client, audio_dir = service_clients

    response = ml_client.post(
        "/v2/predict/batch",
        json={
            "capture_id": "capture-uncertain",
            "segments": [
                {"segment_index": 0, "segment_id": "uncertain-0", "landmarks": _landmarks(10)},
                {"segment_index": 1, "segment_id": "uncertain-1", "landmarks": _landmarks(11)},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["complete"] is False
    assert body["keywords"] == []
    assert [item["status"] for item in body["results"]] == ["uncertain", "uncertain"]
    assert all(item["keyword"] is None for item in body["results"])
    assert len(model.calls) == 1

    # Forwarding the full ML response still fails closed: no partial phrase or
    # audio may be generated when every segment abstains.  The backend uses its
    # explicit record-again result rather than treating this as a server fault.
    rejected = backend_client.post("/process", json=_backend_capture_payload(body))
    assert rejected.status_code == 200
    assert rejected.json() == {"status": "record_again", "record_again": True}
    assert not list(audio_dir.glob("*.mp3"))


def test_malformed_batch_envelope_configuration_and_health_are_controlled(
    monkeypatch: pytest.MonkeyPatch,
    service_clients,
) -> None:
    """Cheap health remains available while batch validation rejects bad input."""

    ml_client, backend_client, audio_dir = service_clients
    monkeypatch.setattr(ml_app, "get_inference_engine", lambda: pytest.fail("health loaded model"))

    assert ml_client.get("/health").status_code == 200
    assert backend_client.get("/health").status_code == 200

    monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "not-a-boolean")
    invalid_config = backend_client.post("/process", json={"keywords": ["hello"]})
    assert invalid_config.status_code == 500
    assert invalid_config.json() == {
        "detail": "REQUIRE_HTTPS_PUBLIC_BASE_URL must be a boolean flag"
    }
    assert not list(audio_dir.glob("*.mp3"))
    monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "0")

    duplicate_indices = {
        "capture_id": "bad-envelope",
        "segments": [
            {"segment_index": 0, "landmarks": _landmarks(20)},
            {"segment_index": 0, "landmarks": _landmarks(21)},
        ],
    }
    duplicate = ml_client.post("/v2/predict/batch", json=duplicate_indices)
    assert duplicate.status_code == 422
    assert duplicate.json() == {"detail": "segment_index values must be unique"}

    too_many_frames = {
        "capture_id": "too-large",
        "segments": [
            {"segment_index": 0, "landmarks": _landmarks(30) * 40},
        ],
    }
    oversized = ml_client.post("/v2/predict/batch", json=too_many_frames)
    assert oversized.status_code == 422
    assert oversized.json() == {"detail": "batch exceeds 450 total raw frames"}
