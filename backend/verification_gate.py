"""BE-M13 backend pre-deployment verification gate.

Default execution is safe for local CI: it runs the backend mock suite and the
cross-service loopback suite with cloud providers disabled. ``--live`` adds
credential-backed Gemini, TTS, and HTTP checks. The live path intentionally
reports only check names, status, and latency; it never logs prompts, response
text, credentials, or raw provider exceptions.

Run from the repository root with::

    python -m backend.verification_gate
    python -m backend.verification_gate --live --base-url https://backend.example
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
import wave
from io import BytesIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTTP_TIMEOUT_SECONDS = 20.0
MP3_MAX_BYTES = 10 * 1024 * 1024
SUPPORTED_SCENARIOS = (
    ("question", ["pay", "medicine", "?"]),
    ("greeting", ["hello"]),
    ("request", ["need", "water"]),
    ("scheduling-question", ["doctor", "tomorrow", "?"]),
    ("urgent", ["emergency", "hospital"]),
)


@dataclass(frozen=True)
class CheckResult:
    """Privacy-safe result of one gate check."""

    name: str
    passed: bool
    latency_ms: float
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.message:
            result["message"] = self.message
        return result


@dataclass(frozen=True)
class GateReport:
    """Complete gate report without private request/response payloads."""

    mode: str
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
        }


def _timed(name: str, operation) -> CheckResult:
    started = time.perf_counter()
    try:
        operation()
    except Exception:
        return CheckResult(
            name=name,
            passed=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            message="check failed; inspect the underlying test/provider configuration",
        )
    return CheckResult(
        name=name,
        passed=True,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def _run_pytest_check(name: str, args: list[str], *, timeout: float = 180.0) -> CheckResult:
    def run() -> None:
        environment = os.environ.copy()
        environment["MOCK_EXTERNAL_SERVICES"] = "1"
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", *args],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            # Do not include captured output: it can contain test payloads or
            # environment-dependent provider details.
            raise RuntimeError(f"pytest exited with {completed.returncode}")

    return _timed(name, run)


def _is_mp3(data: bytes) -> bool:
    """Recognize common MP3 ID3 or MPEG-frame headers without decoding audio."""

    if not data or len(data) > MP3_MAX_BYTES:
        return False
    if data.startswith(b"ID3"):
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _assert_docs() -> None:
    required = {
        REPO_ROOT / "shared" / "contracts.md": (
            "hindi_keywords", "hindi_sentence", "audio_segments", "playback_order",
            "record_again", "en-IN", "hi-IN",
        ),
        REPO_ROOT / "backend" / "README.md": (
            "test_m12_mock_matrix.py", "record_again", "English-then-Hindi", "audio/mpeg",
        ),
        REPO_ROOT / "mobile-handoff.md": (
            "audio_segments", "playback_order", "record_again", "en-IN", "hi-IN",
        ),
    }
    for path, terms in required.items():
        content = path.read_text(encoding="utf-8")
        if any(term not in content for term in terms):
            raise ValueError("required contract documentation is incomplete")


def _make_live_wav_base64() -> str:
    stream = BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x01\x00" * 160)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any], str]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MP3_MAX_BYTES)
            content_type = response.headers.get("Content-Type", "")
            decoded = json.loads(raw.decode("utf-8")) if raw else {}
            return response.status, decoded, content_type
    except HTTPError as error:
        raise RuntimeError(f"HTTP status {error.code}") from None
    except (URLError, ValueError, UnicodeError, json.JSONDecodeError):
        raise RuntimeError("HTTP check failed") from None


def _fetch_audio(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("audio URL is not absolute HTTP(S)")
    request = Request(url, headers={"Accept": "audio/mpeg"}, method="GET")
    try:
        with urlopen(request, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise RuntimeError("audio URL did not return HTTP 200")
            content_type = response.headers.get("Content-Type", "")
            data = response.read(MP3_MAX_BYTES)
    except (HTTPError, URLError):
        raise RuntimeError("audio URL could not be fetched") from None
    if not content_type.lower().startswith("audio/mpeg") or not _is_mp3(data):
        raise ValueError("audio URL did not return playable MP3 bytes")


def _run_live_provider_checks() -> list[CheckResult]:
    from backend.services.gemini_client import keywords_to_bilingual_text
    from backend.services.tts_client import text_to_speech_segments

    checks: list[CheckResult] = []
    if os.getenv("MOCK_EXTERNAL_SERVICES", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return [
            CheckResult(
                "live-configuration",
                False,
                0.0,
                "MOCK_EXTERNAL_SERVICES must be disabled for --live",
            )
        ]
    if not os.getenv("GEMINI_API_KEY", "").strip():
        return [
            CheckResult(
                "live-configuration",
                False,
                0.0,
                "GEMINI_API_KEY is required for --live",
            )
        ]

    for scenario_name, keywords in SUPPORTED_SCENARIOS:
        checks.append(
            _timed(
                f"live-gemini-{scenario_name}",
                lambda keywords=keywords: _check_gemini_result(
                    keywords_to_bilingual_text(keywords)
                ),
            )
        )

    checks.append(
        _timed(
            "live-tts-en-hi",
            lambda: _check_tts_segments(
                text_to_speech_segments(
                    "Can I pay for this medicine?",
                    "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?",
                )
            ),
        )
    )
    return checks


def _check_gemini_result(result: Any) -> None:
    if not (
        isinstance(result.english_sentence, str)
        and result.english_sentence.strip()
        and isinstance(result.hindi_sentence, str)
        and result.hindi_sentence.strip()
        and isinstance(result.hindi_keywords, list)
        and result.hindi_keywords
    ):
        raise ValueError("Gemini result is incomplete")


def _check_tts_segments(segments: Iterable[Any]) -> None:
    segments = tuple(segments)
    if len(segments) != 2:
        raise ValueError("TTS did not return two segments")
    if [segment.language for segment in segments] != ["en-IN", "hi-IN"]:
        raise ValueError("TTS language order is invalid")
    if any(not _is_mp3(segment.audio_content) for segment in segments):
        raise ValueError("TTS returned non-playable MP3 bytes")


def _run_live_http_checks(base_url: str) -> list[CheckResult]:
    base_url = base_url.rstrip("/") + "/"

    def health() -> None:
        status, body, _ = _http_json("GET", urljoin(base_url, "health"))
        if status != 200 or body != {"status": "ok"}:
            raise ValueError("health check failed")

    def process() -> None:
        status, body, _ = _http_json(
            "POST",
            urljoin(base_url, "process"),
            {"keywords": ["pay", "medicine", "?"]},
        )
        if status != 200:
            raise ValueError("process check failed")
        if body.get("subtitles") != [
            {"language": "en-IN", "text": body.get("sentence")},
            {"language": "hi-IN", "text": body.get("hindi_sentence")},
        ]:
            raise ValueError("process subtitle contract failed")
        if body.get("playback_order") != ["en-IN", "hi-IN"]:
            raise ValueError("process playback order failed")
        segments = body.get("audio_segments")
        if not isinstance(segments, list) or [
            segment.get("language") for segment in segments
        ] != ["en-IN", "hi-IN"]:
            raise ValueError("process audio segment contract failed")
        if body.get("audio_url") != segments[0].get("audio_url"):
            raise ValueError("legacy English audio URL was not preserved")
        for segment in segments:
            _fetch_audio(segment["audio_url"])

    def session() -> None:
        status, body, _ = _http_json("POST", urljoin(base_url, "session/create"))
        room_id = body.get("room_id") if status == 200 else None
        if not isinstance(room_id, str) or not room_id:
            raise ValueError("session creation failed")
        status, _, _ = _http_json(
            "POST",
            urljoin(base_url, f"session/{room_id}/message"),
            {"sender": "signer", "text": "M13 verification"},
        )
        if status != 200:
            raise ValueError("session message failed")
        status, body, _ = _http_json(
            "GET", urljoin(base_url, f"session/{room_id}/messages")
        )
        if status != 200 or not body.get("messages"):
            raise ValueError("session read failed")

    def transcribe() -> None:
        status, body, _ = _http_json(
            "POST",
            urljoin(base_url, "transcribe"),
            {"audio": _make_live_wav_base64()},
        )
        if status != 200 or not isinstance(body.get("text"), str):
            raise ValueError("transcribe check failed")

    def audio_missing() -> None:
        status, _, _ = _http_json(
            "GET", urljoin(base_url, "audio/00000000000000000000000000000000.mp3")
        )
        if status != 404:
            raise ValueError("audio missing-file check failed")

    def docs() -> None:
        _assert_docs()

    return [
        _timed("live-http-health", health),
        _timed("live-http-process-and-audio", process),
        _timed("live-http-transcribe", transcribe),
        _timed("live-http-session", session),
        _timed("live-http-audio-missing-file", audio_missing),
        _timed("contract-documentation", docs),
    ]


def run_gate(*, live: bool = False, base_url: str | None = None) -> GateReport:
    """Run the M13 gate and return a privacy-safe structured report."""

    checks = [
        _run_pytest_check("backend-mock-suite", ["backend/tests", "-q"]),
        _run_pytest_check(
            "cross-service-loopback-suite",
            ["tests/test_http_integration.py", "-q"],
        ),
    ]
    if live:
        checks.extend(_run_live_provider_checks())
        checks.extend(
            _run_live_http_checks(
                base_url or os.getenv("SIGNBRIDGE_BACKEND_URL", "http://127.0.0.1:8081")
            )
        )
    else:
        checks.append(_timed("contract-documentation", _assert_docs))
    return GateReport("live" if live else "mock", tuple(checks))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="also run credential-backed provider and HTTP checks",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="live backend base URL; defaults to SIGNBRIDGE_BACKEND_URL or localhost:8081",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the privacy-safe report as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_gate(live=args.live, base_url=args.base_url)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"BE-M13 mode={report.mode} passed={report.passed}")
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            suffix = f" ({check.message})" if check.message else ""
            print(f"{status} {check.name}: {check.latency_ms:.2f} ms{suffix}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
