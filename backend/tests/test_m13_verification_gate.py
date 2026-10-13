"""Unit tests for the privacy-safe BE-M13 verification gate."""

from types import SimpleNamespace
from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

try:
    from backend import verification_gate as gate
except ModuleNotFoundError:
    import verification_gate as gate


def test_mp3_header_check_accepts_id3_and_mpeg_frame() -> None:
    assert gate._is_mp3(b"ID3" + b"payload")
    assert gate._is_mp3(bytes.fromhex("FF FB 90 64") + b"payload")
    assert not gate._is_mp3(b"")
    assert not gate._is_mp3(b"not audio")


def test_tts_gate_requires_ordered_playable_both_language_segments() -> None:
    segment = SimpleNamespace
    valid = (
        segment(language="en-IN", audio_content=b"ID3english"),
        segment(language="hi-IN", audio_content=b"ID3hindi"),
    )
    gate._check_tts_segments(valid)

    with pytest.raises(ValueError):
        gate._check_tts_segments(tuple(reversed(valid)))


def test_gate_report_contains_only_safe_check_metadata() -> None:
    report = gate.GateReport(
        "mock",
        (gate.CheckResult("example", True, 12.345),),
    )

    assert report.passed is True
    assert report.as_dict() == {
        "mode": "mock",
        "passed": True,
        "checks": [{"name": "example", "passed": True, "latency_ms": 12.35}],
    }


def test_failed_subprocess_does_not_return_captured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    private_output = "GEMINI_API_KEY=secret prompt=private text"

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout=private_output, stderr=private_output)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    result = gate._run_pytest_check("mock", ["backend/tests"])

    assert result.passed is False
    assert private_output not in result.message
    assert result.message == "check failed; inspect the underlying test/provider configuration"


def test_default_gate_runs_mock_and_loopback_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_pytest(name, args, timeout=180.0):
        calls.append((name, args))
        return gate.CheckResult(name, True, 1.0)

    monkeypatch.setattr(gate, "_run_pytest_check", fake_pytest)
    monkeypatch.setattr(gate, "_assert_docs", lambda: None)

    report = gate.run_gate(live=False)

    assert report.passed is True
    assert [name for name, _ in calls] == [
        "backend-mock-suite", "cross-service-loopback-suite"
    ]
    assert [check.name for check in report.checks] == [
        "backend-mock-suite", "cross-service-loopback-suite", "contract-documentation"
    ]


def test_live_gate_requires_real_mode_and_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    checks = gate._run_live_provider_checks()

    assert len(checks) == 1
    assert checks[0].name == "live-configuration"
    assert checks[0].passed is False
    assert "secret" not in checks[0].message


def test_live_gate_accepts_vertex_adc_without_api_key(monkeypatch):
    from backend.services import gemini_client, tts_client

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_USE_VERTEX_AI", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_client, "keywords_to_bilingual_text", lambda words: SimpleNamespace(
        english_sentence="Hello.", hindi_sentence="नमस्ते।", hindi_keywords=["नमस्ते"],
    ))
    monkeypatch.setattr(tts_client, "text_to_speech_segments", lambda *args: (
        SimpleNamespace(language="en-IN", audio_content=b"ID3english"),
        SimpleNamespace(language="hi-IN", audio_content=b"ID3hindi"),
    ))
    checks = gate._run_live_provider_checks()
    assert len(checks) == 6
    assert all(check.passed for check in checks)


def test_live_vertex_gate_requires_project(monkeypatch):
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_USE_VERTEX_AI", "1")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    checks = gate._run_live_provider_checks()
    assert len(checks) == 1
    assert not checks[0].passed
    assert "GOOGLE_CLOUD_PROJECT" in checks[0].message


def test_http_json_reads_and_closes_expected_http_error(monkeypatch):
    body = BytesIO(b'{"detail":"Not found"}')
    error = HTTPError("https://example.invalid/audio/missing.mp3", 404,
                      "private reason", {"Content-Type": "application/json"}, body)

    def fake_urlopen(*args, **kwargs):
        raise error

    monkeypatch.setattr(gate, "urlopen", fake_urlopen)
    assert gate._http_json("GET", error.url, expected_status=404) == (
        404, {"detail": "Not found"}, "application/json",
    )
    assert body.closed


@pytest.mark.parametrize("status,options", [(404, {}), (500, {"expected_status": 404})])
def test_http_json_rejects_and_closes_unexpected_http_error(monkeypatch, status, options):
    body = BytesIO(b'{"detail":"private body"}')
    error = HTTPError("https://example.invalid/private-url", status,
                      "private reason", {"Content-Type": "application/json"}, body)

    def fake_urlopen(*args, **kwargs):
        raise error

    monkeypatch.setattr(gate, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match=f"^HTTP status {status}$"):
        gate._http_json("GET", error.url, **options)
    assert body.closed


@pytest.mark.parametrize("raw", [b"private invalid JSON", b"\xff"])
def test_http_json_sanitizes_invalid_expected_error_body(monkeypatch, raw):
    body = BytesIO(raw)
    error = HTTPError("https://example.invalid/private-url", 404,
                      "private reason", {}, body)

    def fake_urlopen(*args, **kwargs):
        raise error

    monkeypatch.setattr(gate, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="^HTTP check failed$"):
        gate._http_json("GET", error.url, expected_status=404)
    assert body.closed


@pytest.mark.parametrize("missing_status", [404, 200, 500])
def test_live_http_gate_missing_audio_with_real_http_error(monkeypatch, missing_status):
    streams = []
    english_url = "https://example.invalid/audio/en.mp3"
    hindi_url = "https://example.invalid/audio/hi.mp3"
    responses = {
        "/health": {"status": "ok"},
        "/process": {
            "sentence": "Hello", "hindi_sentence": "नमस्ते",
            "subtitles": [{"language": "en-IN", "text": "Hello"},
                          {"language": "hi-IN", "text": "नमस्ते"}],
            "playback_order": ["en-IN", "hi-IN"],
            "audio_url": english_url,
            "audio_segments": [{"language": "en-IN", "audio_url": english_url},
                               {"language": "hi-IN", "audio_url": hindi_url}],
        },
        "/transcribe": {"text": "Hello"},
        "/session/create": {"room_id": "test-room"},
        "/session/test-room/message": {},
        "/session/test-room/messages": {"messages": [{"text": "Hello"}]},
    }

    def fake_urlopen(request, **kwargs):
        path = gate.urlsplit(request.full_url).path
        if path == "/audio/00000000000000000000000000000000.mp3":
            stream = BytesIO(b'{"detail":"private missing body"}')
            streams.append(stream)
            if missing_status != 200:
                raise HTTPError(request.full_url, missing_status, "private reason",
                                {"Content-Type": "application/json"}, stream)
        elif path in {"/audio/en.mp3", "/audio/hi.mp3"}:
            stream = BytesIO(b"ID3mock")
            streams.append(stream)
        else:
            stream = BytesIO(json.dumps(responses[path]).encode("utf-8"))
            streams.append(stream)
        stream.status = 200
        stream.headers = {"Content-Type": "audio/mpeg" if path in {
            "/audio/en.mp3", "/audio/hi.mp3",
        } else "application/json"}
        return stream

    monkeypatch.setattr(gate, "urlopen", fake_urlopen)
    monkeypatch.setattr(gate, "_assert_docs", lambda: None)
    checks = gate._run_live_http_checks("https://example.invalid")
    missing = next(check for check in checks if check.name == "live-http-audio-missing-file")
    assert missing.passed is (missing_status == 404)
    assert all(check.passed for check in checks if check is not missing)
    assert all(stream.closed for stream in streams)
    assert "private" not in json.dumps(gate.GateReport("live", tuple(checks)).as_dict())
