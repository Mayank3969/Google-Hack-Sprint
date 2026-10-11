"""Mocked cross-service SignBridge vertical-slice integration test."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import sys
from urllib.parse import urlsplit
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ml-service"))

from backend.main import app as backend_app  # noqa: E402
from backend.fixtures.mock_audio import MOCK_MP3_BYTES  # noqa: E402
from backend.routes import process as process_route  # noqa: E402
from backend.services import firestore_client, stt_client  # noqa: E402
from service import app as ml_app_module  # noqa: E402


class FakePrediction:
    def __init__(self, label):
        self.label = label
        self.top_k = [{"label": label, "class_id": 0, "confidence": 1.0}]


class FakeMLEngine:
    def __init__(self, labels):
        self.labels = iter(labels)

    def predict_details(self, landmarks, top_k=1):
        assert landmarks.shape == (3, 110, 3)
        assert landmarks.dtype == np.float32
        assert np.isfinite(landmarks).all()
        assert top_k == 1
        return FakePrediction(next(self.labels))


def _wav_base64() -> str:
    stream = BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x01\x00" * 160)
    return base64.b64encode(stream.getvalue()).decode("ascii")


@pytest.mark.parametrize(
    "sign_labels, expected_sentence",
    [(("pay",), "Pay."), (("pay", "medicine"), "Pay medicine.")],
    ids=["isolated-sign", "client-accumulates-multiple-signs"],
)
def test_ml_to_backend_to_session_vertical_slice(
    monkeypatch, tmp_path, sign_labels, expected_sentence
):
    engine = FakeMLEngine(sign_labels)
    monkeypatch.setattr(ml_app_module, "get_engine", lambda: engine)
    monkeypatch.setattr(process_route, "AUDIO_DIR", tmp_path)
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "0")
    monkeypatch.setenv("TTS_VOICE_GENDER", "NEUTRAL")
    monkeypatch.setenv("TTS_LANGUAGE_CODE", "en-IN")
    monkeypatch.setenv("TTS_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("STT_LANGUAGE_CODE", "en-IN")
    monkeypatch.setenv("STT_TIMEOUT_SECONDS", "10")
    firestore_client.reset_mock_repository()

    landmarks = np.zeros((3, 110, 3), dtype=np.float32).tolist()
    with TestClient(ml_app_module.app) as ml_client, TestClient(backend_app) as backend_client:
        for client in (ml_client, backend_client):
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

        # The mobile client accumulates isolated predictions; the backend only
        # receives the resulting keyword list and never orchestrates ML calls.
        keywords = []
        for expected_label in sign_labels:
            predicted = ml_client.post("/predict", json={"landmarks": landmarks})
            assert predicted.status_code == 200
            assert predicted.json() == {"keywords": [expected_label]}
            keywords.extend(predicted.json()["keywords"])
        assert keywords == list(sign_labels)

        processed = backend_client.post("/process", json={"keywords": keywords})
        assert processed.status_code == 200
        process_body = processed.json()
        assert process_body["sentence"] == expected_sentence
        audio_url = urlsplit(process_body["audio_url"])
        assert audio_url.scheme == "http" and audio_url.netloc == "testserver"
        assert audio_url.path.startswith("/audio/") and audio_url.path.endswith(".mp3")
        audio = backend_client.get(process_body["audio_url"])
        assert audio.status_code == 200
        assert audio.headers["content-type"].startswith("audio/mpeg")
        assert audio.content == MOCK_MP3_BYTES

        room = backend_client.post("/session/create")
        assert room.status_code == 200
        room_id = room.json()["room_id"]
        empty = backend_client.get(f"/session/{room_id}/messages")
        assert empty.status_code == 200
        assert empty.json() == {"messages": []}
        started = datetime.now(timezone.utc)
        saved_signer = backend_client.post(
            f"/session/{room_id}/message",
            json={"sender": "signer", "text": process_body["sentence"]},
        )
        assert saved_signer.status_code == 200
        assert saved_signer.json() == {"status": "ok"}

        transcribed = backend_client.post("/transcribe", json={"audio": _wav_base64()})
        assert transcribed.status_code == 200
        assert transcribed.json() == {"text": stt_client.MOCK_TRANSCRIPT}
        saved_speaker = backend_client.post(
            f"/session/{room_id}/message",
            json={"sender": "speaker", "text": transcribed.json()["text"]},
        )
        assert saved_speaker.status_code == 200
        assert saved_speaker.json() == {"status": "ok"}

        messages = backend_client.get(f"/session/{room_id}/messages")
        assert messages.status_code == 200
        items = messages.json()["messages"]
        assert [(item["sender"], item["text"]) for item in items] == [
            ("signer", expected_sentence),
            ("speaker", transcribed.json()["text"]),
        ]
        timestamps = [datetime.fromisoformat(item["timestamp"]) for item in items]
        assert all(item["timestamp"].endswith("Z") for item in items)
        assert all(timestamp.utcoffset() == timedelta(0) for timestamp in timestamps)
        assert timestamps == sorted(timestamps)
        assert all(started <= timestamp <= datetime.now(timezone.utc) for timestamp in timestamps)

        rejected = backend_client.post(
            f"/session/{room_id}/message", json={"sender": "speaker", "text": "   "}
        )
        assert rejected.status_code == 422
        assert backend_client.get(f"/session/{room_id}/messages").json() == messages.json()

        other_room = backend_client.post("/session/create").json()["room_id"]
        assert other_room != room_id
        assert backend_client.get(f"/session/{other_room}/messages").json() == {"messages": []}
        for missing in (
            backend_client.get("/session/missing-room/messages"),
            backend_client.post(
                "/session/missing-room/message", json={"sender": "signer", "text": "hello"}
            ),
        ):
            assert missing.status_code == 404
            assert missing.json() == {"detail": "conversation room not found"}
