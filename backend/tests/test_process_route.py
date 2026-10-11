"""HTTP-level tests for backend module BE-M4."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

try:
    from backend.main import app
    from backend.routes import process as process_route
    from backend.services.gemini_client import GeminiProviderError
    from backend.services.tts_client import TTSProviderError
    from backend.fixtures.mock_audio import MOCK_MP3_BYTES
except ModuleNotFoundError:
    from main import app
    from routes import process as process_route
    from services.gemini_client import GeminiProviderError
    from services.tts_client import TTSProviderError
    from fixtures.mock_audio import MOCK_MP3_BYTES


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
    assert body["sentence"] == "Can I pay medicine?"
    assert body["audio_url"].startswith("http://testserver/audio/")

    audio_path = "/" + body["audio_url"].split("/", 3)[-1]
    audio_response = client.get(audio_path)

    assert audio_response.status_code == 200
    assert audio_response.headers["content-type"].startswith("audio/mpeg")
    assert audio_response.content == MOCK_MP3_BYTES
    assert audio_response.content
    assert list(isolated_audio_dir.glob("*.mp3"))


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
        "keywords_to_sentence",
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
        "keywords_to_sentence",
        lambda keywords: "Hello from SignBridge.",
    )
    monkeypatch.setattr(
        process_route,
        "text_to_speech",
        lambda sentence: (_ for _ in ()).throw(
            TTSProviderError("provider details should not escape")
        ),
    )

    response = client.post("/process", json={"keywords": ["hello"]})

    assert response.status_code == 502
    assert response.json() == {"detail": "speech synthesis failed"}


@pytest.fixture
def forbid_process_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_call(*args, **kwargs):
        pytest.fail("invalid public URL must fail before providers or file writes")

    monkeypatch.setattr(process_route, "keywords_to_sentence", unexpected_call)
    monkeypatch.setattr(process_route, "text_to_speech", unexpected_call)
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
