"""HTTP-level tests for backend modules BE-M4 and BE-M11."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

try:
    from backend.main import app
    from backend.routes import process as process_route
    from backend.services.gemini_client import (
        BilingualText,
        GeminiProviderError,
        SentenceDecision,
    )
    from backend.services.tts_client import SpeechSegment, TTSProviderError
    from backend.fixtures.mock_audio import (
        MOCK_MP3_BYTES_EN_IN,
        MOCK_MP3_BYTES_HI_IN,
    )
except ModuleNotFoundError:
    from main import app
    from routes import process as process_route
    from services.gemini_client import BilingualText, GeminiProviderError, SentenceDecision
    from services.tts_client import SpeechSegment, TTSProviderError
    from fixtures.mock_audio import MOCK_MP3_BYTES_EN_IN, MOCK_MP3_BYTES_HI_IN


client = TestClient(app)


@pytest.fixture
def isolated_audio_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(process_route, "AUDIO_DIR", tmp_path)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "0")
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    return tmp_path


def test_process_returns_sentence_and_retrievable_audio(
    isolated_audio_dir: Path,
) -> None:
    response = client.post(
        "/process",
        json={"keywords": ["pay", "medicine", "?"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sentence"] == "Can I pay for this medicine?"
    assert body["hindi_keywords"] == ["भुगतान", "दवा", "?"]
    assert body["hindi_sentence"] == "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?"
    assert body["subtitles"] == [
        {"language": "en-IN", "text": "Can I pay for this medicine?"},
        {"language": "hi-IN", "text": "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?"},
    ]
    assert body["playback_order"] == ["en-IN", "hi-IN"]
    assert [segment["language"] for segment in body["audio_segments"]] == [
        "en-IN", "hi-IN"
    ]
    assert body["audio_url"].startswith("http://testserver/audio/")
    assert body["audio_url"] == body["audio_segments"][0]["audio_url"]

    audio_path = "/" + body["audio_url"].split("/", 3)[-1]
    audio_response = client.get(audio_path)

    assert audio_response.status_code == 200
    assert audio_response.headers["content-type"].startswith("audio/mpeg")
    assert audio_response.content == MOCK_MP3_BYTES_EN_IN
    assert audio_response.content
    assert list(isolated_audio_dir.glob("*.mp3"))


def test_process_locale_selects_preferred_playback_without_dropping_bilingual_output(
    isolated_audio_dir: Path,
) -> None:
    response = client.post(
        "/process",
        json={"keywords": ["pay", "medicine", "?"], "locale": "hi-IN"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selected_language"] == "hi-IN"
    assert body["selected_sentence"] == body["hindi_sentence"]
    assert body["selected_audio_url"] == body["audio_segments"][1]["audio_url"]
    assert body["playback_order"] == ["hi-IN", "en-IN"]
    assert len(body["subtitles"]) == 2
    assert len(body["audio_segments"]) == 2

    hindi_audio_path = "/" + body["audio_segments"][1]["audio_url"].split("/", 3)[-1]
    hindi_audio_response = client.get(hindi_audio_path)
    assert hindi_audio_response.status_code == 200
    assert hindi_audio_response.headers["content-type"].startswith("audio/mpeg")
    assert hindi_audio_response.content == MOCK_MP3_BYTES_HI_IN


@pytest.mark.parametrize(
    "keywords, expected_sentence",
    [
        (["pay", "medicine", "?"], "Can I pay for this medicine?"),
        (["hello"], "Hello."),
        (["need", "water"], "I need water."),
        (["doctor", "tomorrow", "?"], "Can I see the doctor tomorrow?"),
        (["emergency", "hospital"], "It is an emergency; I need to go to the hospital."),
    ],
    ids=["question", "greeting", "request", "schedule-question", "urgent"],
)
def test_local_mock_process_covers_five_sentence_types(
    isolated_audio_dir: Path,
    keywords: list[str],
    expected_sentence: str,
) -> None:
    """Exercise the deterministic bilingual response and both audio segments."""

    response = client.post("/process", json={"keywords": keywords})

    assert response.status_code == 200
    body = response.json()
    assert body["sentence"] == expected_sentence
    assert body["audio_url"].startswith("http://testserver/audio/")

    audio_path = "/" + body["audio_url"].split("/", 3)[-1]
    audio_response = client.get(audio_path)

    assert audio_response.status_code == 200
    assert audio_response.headers["content-type"].startswith("audio/mpeg")
    assert audio_response.content == MOCK_MP3_BYTES_EN_IN
    assert len(body["audio_segments"]) == 2
    hindi_audio_path = "/" + body["audio_segments"][1]["audio_url"].split("/", 3)[-1]
    hindi_audio_response = client.get(hindi_audio_path)
    assert hindi_audio_response.status_code == 200
    assert hindi_audio_response.content == MOCK_MP3_BYTES_HI_IN


def test_process_requests_a_new_recording_when_sentence_is_incomplete(
    isolated_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown/unrecognized words must not be translated or synthesized."""

    monkeypatch.setattr(
        process_route,
        "text_to_speech_segments",
        lambda *args: pytest.fail("record-again responses must skip TTS"),
    )

    response = client.post("/process", json={"keywords": ["unrecognized-sign"]})

    assert response.status_code == 200
    assert response.json() == {"status": "record_again", "record_again": True}
    assert not list(isolated_audio_dir.glob("*.mp3"))


def test_process_uses_configured_public_base_url(
    isolated_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://backend.example.test/")
    monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "1")

    response = client.post("/process", json={"keywords": ["hello"]})

    assert response.status_code == 200
    assert response.json()["audio_url"].startswith(
        "https://backend.example.test/audio/"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"keywords": []},
        {"keywords": [1]},
        {"keywords": [" "]},
        {"keywords": ["x" * 65]},
        {"keywords": ["word"] * 21},
    ],
)
def test_process_rejects_malformed_keywords(
    isolated_audio_dir: Path,
    payload: dict[str, object],
) -> None:
    response = client.post("/process", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_process_accepts_segmented_ml_results_and_uses_recognized_words_in_index_order(
    isolated_audio_dir: Path,
) -> None:
    response = client.post(
        "/process",
        json={
            "capture_id": "capture-123",
            "results": [
                {
                    "segment_index": 3,
                    "status": "recognized",
                    "keyword": "?",
                    "confidence": 0.88,
                    "coverage": 0.91,
                },
                {
                    "segment_index": 1,
                    "status": "recognized",
                    "keyword": "medicine",
                    "confidence": 0.89,
                    "coverage": 0.95,
                    "top_k": [
                        {"label": "medicine", "confidence": 0.89},
                    ],
                },
                {
                    "segment_index": 0,
                    "status": "recognized",
                    "keyword": "pay",
                    "confidence": 0.91,
                    "coverage": 0.96,
                },
                {
                    "segment_index": 2,
                    "status": "recognized",
                    "keyword": ".",
                    "confidence": 0.88,
                    "coverage": 0.91,
                },
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sentence"] == "Can I pay for this medicine?"
    assert body["hindi_keywords"] == ["भुगतान", "दवा", "?"]
    assert body["playback_order"] == ["en-IN", "hi-IN"]


def test_process_accepts_full_ml_v2_batch_envelope(
    isolated_audio_dir: Path,
) -> None:
    response = client.post(
        "/process",
        json={
            "capture_id": "capture-v2",
            "complete": True,
            "integration_ready": True,
            "keywords": ["pay", "medicine", "?"],
            "warnings": [],
            "results": [
                {
                    "segment_index": 1,
                    "segment_id": "capture-v2-1",
                    "status": "recognized",
                    "keyword": "medicine",
                    "class_id": 5,
                    "confidence": 0.91,
                    "top_k": [
                        {"class_id": 5, "label": "medicine", "confidence": 0.91}
                    ],
                    "quality": {"coordinate_coverage": 0.92},
                    "integration_ready": True,
                },
                {
                    "segment_index": 0,
                    "segment_id": "capture-v2-0",
                    "status": "recognized",
                    "keyword": "pay",
                    "class_id": 2,
                    "confidence": 0.93,
                    "top_k": [
                        {"class_id": 2, "label": "pay", "confidence": 0.93}
                    ],
                    "quality": {"coordinate_coverage": 0.94},
                    "integration_ready": True,
                },
                {
                    "segment_index": 2,
                    "segment_id": "capture-v2-2",
                    "status": "recognized",
                    "keyword": "?",
                    "class_id": 9,
                    "confidence": 0.89,
                    "top_k": [{"class_id": 9, "label": "?", "confidence": 0.89}],
                    "quality": {"coordinate_coverage": 0.9},
                    "integration_ready": True,
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["sentence"] == "Can I pay for this medicine?"


def test_segmented_results_use_keywords_and_preserve_duplicate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIGNBRIDGE_RECOGNITION_THRESHOLD", "0.5")

    payload = process_route.ProcessInput(
        capture_id="capture-top-k",
        results=[
            {
                "segment_index": 1,
                "status": "recognized",
                "keyword": "pay",
                "confidence": 0.95,
                "coverage": 0.95,
                "top_k": [
                    {"label": "medicine", "confidence": 0.90},
                    {"label": "?", "confidence": 0.80},
                ],
            },
            {
                "segment_index": 0,
                "status": "recognized",
                "keyword": "pay",
                "confidence": 0.95,
                "coverage": 0.95,
                "top_k": [
                    {"label": "pay", "confidence": 0.95},
                    {"label": "medicine", "confidence": 0.85},
                    {"label": "PAY", "confidence": 0.80},
                    {"label": "noise", "confidence": 0.49},
                ],
            },
        ],
    )

    assert payload.ordered_keywords() == ["pay", "pay"]


def test_segmented_threshold_applies_to_confidence_and_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIGNBRIDGE_RECOGNITION_THRESHOLD", "0.8")

    payload = process_route.ProcessInput(
        capture_id="capture-threshold",
        results=[
            {
                "segment_index": 0,
                "status": "recognized",
                "keyword": None,
                "confidence": 0.95,
                "coverage": 0.95,
                "top_k": [
                    {"label": "below", "confidence": 0.79},
                    {"label": "at-threshold", "confidence": 0.80},
                ],
            },
            {
                "segment_index": 1,
                "status": "recognized",
                "keyword": "coverage-too-low",
                "confidence": 0.95,
                "coverage": 0.79,
            },
        ],
    )

    assert payload.batch_is_complete() is False
    assert payload.ordered_keywords() == []


def test_segmented_candidates_below_threshold_request_recording_again(
    isolated_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIGNBRIDGE_RECOGNITION_THRESHOLD", "0.8")
    monkeypatch.setattr(
        process_route,
        "text_to_speech_segments",
        lambda *args: pytest.fail("below-threshold candidates must skip TTS"),
    )

    response = client.post(
        "/process",
        json={
            "capture_id": "capture-low-confidence",
            "results": [
                {
                    "segment_index": 0,
                    "status": "recognized",
                    "keyword": None,
                    "confidence": 0.9,
                    "coverage": 0.9,
                    "top_k": [{"label": "pay", "confidence": 0.79}],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "record_again", "record_again": True}
    assert not list(isolated_audio_dir.glob("*.mp3"))


def test_invalid_recognition_threshold_fails_closed(
    isolated_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIGNBRIDGE_RECOGNITION_THRESHOLD", "not-a-number")

    response = client.post(
        "/process",
        json={
            "capture_id": "capture-invalid-threshold",
            "results": [
                {
                    "segment_index": 0,
                    "status": "recognized",
                    "keyword": "hello",
                    "confidence": 0.9,
                    "coverage": 0.9,
                }
            ],
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "SIGNBRIDGE_RECOGNITION_THRESHOLD must be a finite number from 0 to 1"
        )
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "results": [
                {"segment_index": 0, "status": "recognized", "keyword": "pay",
                 "confidence": 0.9, "coverage": 0.9},
            ],
        },
        {"capture_id": "capture-123"},
        {
            "capture_id": "capture-123",
            "results": [
                {"segment_index": 0, "status": "recognized", "keyword": None,
                 "confidence": 0.9, "coverage": 0.9},
                {"segment_index": 0, "status": "recognized", "keyword": "pay",
                 "confidence": 0.9, "coverage": 0.9},
            ],
        },
        {
            "capture_id": "capture-123",
            "results": [
                {"segment_index": 0, "status": "recognized", "keyword": "pay",
                 "confidence": 1.1, "coverage": 0.9},
            ],
        },
        {
            "capture_id": "capture-123",
            "results": [
                {"segment_index": 0, "status": "unknown", "keyword": "pay",
                 "confidence": 0.9, "coverage": 0.9},
            ],
        },
        {
            "capture_id": "capture-123",
            "keywords": ["pay"],
            "results": [
                {"segment_index": 0, "status": "recognized", "keyword": "pay",
                 "confidence": 0.9, "coverage": 0.9},
            ],
        },
    ],
)
def test_process_rejects_invalid_segmented_ml_results(
    isolated_audio_dir: Path,
    payload: dict[str, object],
) -> None:
    response = client.post("/process", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_process_rejects_a_batch_with_any_uncertain_segment_before_providers(
    isolated_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_route,
        "keywords_to_sentence_decision",
        lambda *args: pytest.fail("uncertain batches must skip Gemini"),
    )
    monkeypatch.setattr(
        process_route,
        "text_to_speech_segments",
        lambda *args: pytest.fail("uncertain batches must skip TTS"),
    )

    results = [
        {
            "segment_index": 0,
            "status": "recognized",
            "keyword": "hello",
            "confidence": 0.95,
            "coverage": 0.95,
        }
    ] + [
        {
            "segment_index": 1,
            "status": "uncertain",
            "keyword": None,
            "confidence": 0.2,
            "coverage": 0.4,
        }
    ]

    response = client.post(
        "/process", json={"capture_id": "capture-10", "results": results}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "record_again", "record_again": True}
    assert not list(isolated_audio_dir.glob("*.mp3"))


def test_missing_audio_returns_404(isolated_audio_dir: Path) -> None:
    response = client.get("/audio/00000000000000000000000000000000.mp3")

    assert response.status_code == 404
    assert response.json() == {"detail": "audio not found"}


def test_invalid_audio_id_cannot_escape_audio_directory(
    isolated_audio_dir: Path,
) -> None:
    response = client.get("/audio/../secrets.mp3")

    assert response.status_code in {404, 405}


def test_gemini_failure_returns_controlled_502(
    isolated_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_route,
        "keywords_to_sentence_decision",
        lambda keywords: (_ for _ in ()).throw(
            GeminiProviderError("provider details should not escape")
        ),
    )

    response = client.post("/process", json={"keywords": ["hello"]})

    assert response.status_code == 502
    assert response.json() == {"detail": "sentence generation failed"}


def test_tts_failure_returns_controlled_502(
    isolated_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_route,
        "keywords_to_sentence_decision",
        lambda keywords: SentenceDecision(
            True,
            BilingualText(["नमस्ते"], "Hello from SignBridge.", "साइनब्रिज से नमस्ते।"),
        ),
    )
    monkeypatch.setattr(
        process_route,
        "text_to_speech_segments",
        lambda english, hindi: (_ for _ in ()).throw(
            TTSProviderError("provider details should not escape")
        ),
    )

    response = client.post("/process", json={"keywords": ["hello"]})

    assert response.status_code == 502
    assert response.json() == {"detail": "speech synthesis failed"}
    assert not list(isolated_audio_dir.glob("*.mp3"))


def test_bilingual_tts_failure_after_provider_work_never_publishes_partial_audio(
    isolated_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_route,
        "keywords_to_sentence_decision",
        lambda keywords: SentenceDecision(
            True,
            BilingualText(["नमस्ते"], "Hello from SignBridge.", "साइनब्रिज से नमस्ते।"),
        ),
    )

    def failing_segments(english: str, hindi: str):
        raise TTSProviderError("Hindi provider details should not escape")

    monkeypatch.setattr(process_route, "text_to_speech_segments", failing_segments)

    response = client.post("/process", json={"keywords": ["hello"]})

    assert response.status_code == 502
    assert response.json() == {"detail": "speech synthesis failed"}
    assert not list(isolated_audio_dir.glob("*.mp3"))


def test_storage_failure_cleans_already_written_segment(
    isolated_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write_bytes = Path.write_bytes
    writes = {"count": 0}

    def fail_on_second_write(path: Path, content: bytes) -> int:
        writes["count"] += 1
        if writes["count"] == 2:
            raise OSError("disk full")
        return original_write_bytes(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_on_second_write)

    response = client.post("/process", json={"keywords": ["hello"]})

    assert response.status_code == 500
    assert response.json() == {"detail": "audio storage failed"}
    assert not list(isolated_audio_dir.glob("*.mp3"))


@pytest.fixture
def forbid_process_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_call(*args, **kwargs):
        pytest.fail("invalid public URL must fail before providers or file writes")

    monkeypatch.setattr(process_route, "keywords_to_sentence_decision", unexpected_call)
    monkeypatch.setattr(process_route, "text_to_speech_segments", unexpected_call)
    monkeypatch.setattr(Path, "write_bytes", unexpected_call)


@pytest.mark.parametrize(
    "base_url",
    [
        "relative/path", "ftp://backend.example.test", "https://",
        "https://[broken", "https://[::1]junk", "https://backend.example.test:bad",
        "https://backend.example.test:65536", "https://backend.example.test:0",
        "https://backend.example.test:", "https://user:secret@backend.example.test",
        "https://backend.example.test?token=secret", "https://backend.example.test#audio",
        "https://backend.example.test?", "https://backend.example.test#",
        "https://bad host", "https://bad\\host", "https://bad\nhost",
        "https://backend.example.test\x7f", "https://bad%20host",
        "https://bad..host", "https://bad.host..", "https://-bad.host", " ",
    ],
)
def test_invalid_public_url_is_controlled_before_side_effects(
    isolated_audio_dir, forbid_process_side_effects, monkeypatch, base_url,
) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", base_url)
    response = client.post("/process", json={"keywords": ["hello"]})
    assert response.status_code == 500
    assert response.json() == {
        "detail": "PUBLIC_BASE_URL must be an absolute HTTP(S) URL without credentials, query, or fragment"
    }
    assert not list(isolated_audio_dir.iterdir())


@pytest.mark.parametrize("cloud_run", [False, True])
def test_deployed_http_url_is_rejected_before_side_effects(
    isolated_audio_dir, forbid_process_side_effects, monkeypatch, cloud_run,
) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://backend.example.test")
    if cloud_run:
        monkeypatch.setenv("K_SERVICE", "signbridge-backend")
    else:
        monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "1")
    response = client.post("/process", json={"keywords": ["hello"]})
    assert response.status_code == 500
    assert response.json() == {
        "detail": "PUBLIC_BASE_URL must use HTTPS in deployed environments"
    }


@pytest.mark.parametrize("base_url", [None, "", " "])
def test_deployed_url_cannot_fall_back_to_request_host(
    isolated_audio_dir, forbid_process_side_effects, monkeypatch, base_url,
) -> None:
    monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "1")
    if base_url is not None:
        monkeypatch.setenv("PUBLIC_BASE_URL", base_url)
    with TestClient(app, base_url="https://untrusted-host.test") as https_client:
        response = https_client.post("/process", json={"keywords": ["hello"]})
    assert response.status_code == 500
    assert response.json() == {
        "detail": "PUBLIC_BASE_URL must be configured in deployed environments"
    }


@pytest.mark.parametrize(
    "base_url", ["http://localhost:8081/", "http://[::1]:8081/", "https://backend.example.test/api/"]
)
def test_valid_local_and_prefixed_public_urls(isolated_audio_dir, monkeypatch, base_url) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", base_url)
    response = client.post("/process", json={"keywords": ["hello"]})
    assert response.status_code == 200
    assert response.json()["audio_url"].startswith(base_url.rstrip("/") + "/audio/")


def test_invalid_https_flag_fails_closed(
    isolated_audio_dir, forbid_process_side_effects, monkeypatch,
) -> None:
    monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "tru")
    response = client.post("/process", json={"keywords": ["hello"]})
    assert response.status_code == 500
    assert response.json() == {"detail": "REQUIRE_HTTPS_PUBLIC_BASE_URL must be a boolean flag"}
