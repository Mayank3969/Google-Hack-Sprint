"""HTTP-level tests for backend module BE-M6."""

from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace
import wave

import pytest
from fastapi.testclient import TestClient

try:
    from backend.main import app
    from backend.routes import transcribe as transcribe_route
    from backend.routes.transcribe_request import MAX_TRANSCRIBE_BODY_BYTES
    from backend.services.audio_validation import MAX_AUDIO_BYTES
    from backend.services import stt_client
except ModuleNotFoundError:
    from main import app
    from routes import transcribe as transcribe_route
    from routes.transcribe_request import MAX_TRANSCRIBE_BODY_BYTES
    from services.audio_validation import MAX_AUDIO_BYTES
    from services import stt_client


client = TestClient(app)


def make_wav(*, frames: bytes) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(frames)
    return buffer.getvalue()


def encoded(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")


@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")


def test_transcribe_valid_wav_returns_mock_text_over_http() -> None:
    audio = make_wav(frames=b"\x01\x00" * 160)

    response = client.post("/transcribe", json={"audio": encoded(audio)})

    assert response.status_code == 200
    assert response.json() == {"text": stt_client.MOCK_TRANSCRIPT}
    assert b"RIFF" not in response.content


def test_transcribe_silence_returns_empty_text() -> None:
    audio = make_wav(frames=b"\x00\x00" * 160)

    response = client.post("/transcribe", json={"audio": encoded(audio)})

    assert response.status_code == 200
    assert response.json() == {"text": ""}


@pytest.mark.parametrize(
    "payload, expected_status, expected_detail",
    [
        (
            {"audio": "not-base64***"},
            400,
            "audio must be valid base64",
        ),
        (
            {"audio": ""},
            400,
            "audio must not be empty",
        ),
        (
            {"audio": encoded(b"not a wav")},
            400,
            "audio must be a valid mono 16000 Hz 16-bit PCM WAV of at most 30 seconds",
        ),
    ],
)
def test_transcribe_rejects_invalid_audio(
    payload: dict[str, str],
    expected_status: int,
    expected_detail: str,
) -> None:
    response = client.post("/transcribe", json=payload)

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_transcribe_rejects_decoded_audio_over_limit() -> None:
    oversized_audio = b"x" * (MAX_AUDIO_BYTES + 1)

    response = client.post("/transcribe", json={"audio": encoded(oversized_audio)})

    assert response.status_code == 413
    assert response.json() == {"detail": "audio exceeds the maximum size"}


def test_transcribe_rejects_base64_character_limit_before_decoding() -> None:
    too_many_characters = "A" * (transcribe_route.MAX_BASE64_AUDIO_CHARS + 1)

    response = client.post("/transcribe", json={"audio": too_many_characters})

    assert response.status_code == 413
    assert response.json() == {"detail": "audio exceeds the maximum size"}


def test_transcribe_rejects_oversized_raw_request_before_json_parsing() -> None:
    oversized_body = b"{" + b"x" * MAX_TRANSCRIBE_BODY_BYTES

    response = client.post(
        "/transcribe",
        content=oversized_body,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "request body exceeds the maximum size"
    }


def test_transcribe_rejects_missing_audio_without_echoing_validation_details() -> None:
    response = client.post("/transcribe", json={})

    assert response.status_code == 422
    assert response.json() == {
        "detail": "request must contain an audio string in valid JSON"
    }


def test_transcribe_rejects_invalid_json_with_safe_error() -> None:
    response = client.post(
        "/transcribe",
        content=b"{not-json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "request must contain an audio string in valid JSON"
    }


def test_transcribe_requires_json_content_type() -> None:
    response = client.post(
        "/transcribe",
        content=b"{}",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 415
    assert response.json() == {"detail": "request Content-Type must be JSON"}


def test_transcribe_rejects_non_identity_content_encoding() -> None:
    response = client.post(
        "/transcribe",
        json={"audio": encoded(b"audio")},
        headers={"Content-Encoding": "gzip"},
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "request Content-Encoding must be identity"
    }


def test_transcribe_maps_provider_failure_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = make_wav(frames=b"\x01\x00" * 160)
    monkeypatch.setattr(
        transcribe_route.stt_client,
        "speech_to_text",
        lambda raw: (_ for _ in ()).throw(
            stt_client.STTProviderError("provider details should not escape")
        ),
    )

    response = client.post("/transcribe", json={"audio": encoded(audio)})

    assert response.status_code == 502
    assert response.json() == {"detail": "speech transcription failed"}


def test_transcribe_maps_non_string_provider_result_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = make_wav(frames=b"\x01\x00" * 160)
    monkeypatch.setattr(
        transcribe_route.stt_client,
        "speech_to_text",
        lambda raw: SimpleNamespace(text="not-a-string-result"),
    )

    response = client.post("/transcribe", json={"audio": encoded(audio)})

    assert response.status_code == 502
    assert response.json() == {"detail": "speech transcription failed"}
