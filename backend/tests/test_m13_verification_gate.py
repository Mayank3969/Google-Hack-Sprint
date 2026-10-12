"""Unit tests for the privacy-safe BE-M13 verification gate."""

from types import SimpleNamespace

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
