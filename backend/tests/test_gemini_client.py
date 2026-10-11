"""Deterministic tests for backend module BE-M2."""

from types import SimpleNamespace

import pytest

try:
    from backend.services import gemini_client
except ModuleNotFoundError:
    from services import gemini_client


@pytest.fixture(autouse=True)
def default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)


def test_mock_mode_returns_a_deterministic_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = gemini_client.keywords_to_sentence(["pay", "medicine", "?"])

    assert result == "Can I pay medicine?"
    assert "pay" in result.lower()
    assert "medicine" in result.lower()


def test_real_mode_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(gemini_client.GeminiConfigurationError, match="GEMINI_API_KEY"):
        gemini_client.keywords_to_sentence(["hello"])


@pytest.mark.parametrize(
    "keywords",
    [
        [],
        ["   "],
        ["valid", 3],
        ["x" * (gemini_client.MAX_KEYWORD_LENGTH + 1)],
        ["word"] * (gemini_client.MAX_KEYWORDS + 1),
    ],
)
def test_keyword_validation_rejects_unsafe_input(
    monkeypatch: pytest.MonkeyPatch, keywords: list[object]
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")

    with pytest.raises(ValueError):
        gemini_client.keywords_to_sentence(keywords)  # type: ignore[arg-type]


def test_real_path_uses_configured_model_and_strips_code_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def generate_content(self, prompt: str, *, request_options: dict) -> SimpleNamespace:
            captured["prompt"] = prompt
            captured["request_options"] = request_options
            return SimpleNamespace(text="```text\nCan I pay for this medicine?\n```")

    def fake_create_model(model_name: str, api_key: str) -> FakeModel:
        captured["model_name"] = model_name
        captured["api_key"] = api_key
        return FakeModel()

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setenv("GEMINI_MODEL", "configured-test-model")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setattr(gemini_client, "_create_real_model", fake_create_model)

    result = gemini_client.keywords_to_sentence(["pay", "medicine", "?"])

    assert result == "Can I pay for this medicine?"
    assert captured["model_name"] == "configured-test-model"
    assert captured["api_key"] == "test-key-not-a-credential"
    assert "pay" in captured["prompt"]
    assert "medicine" in captured["prompt"]
    assert captured["request_options"] == {"timeout": 12.5, "retry": None}


def test_provider_failure_has_a_stable_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "test-key-not-a-credential"

    class FailingModel:
        def generate_content(self, prompt: str, *, request_options: dict) -> None:
            raise RuntimeError("provider details should not escape")

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setattr(
        gemini_client,
        "_create_real_model",
        lambda model_name, api_key: FailingModel(),
    )

    with pytest.raises(gemini_client.GeminiProviderError) as error:
        gemini_client.keywords_to_sentence(["hello"])

    assert str(error.value) == "Gemini provider request failed"
    assert secret not in str(error.value)


def test_empty_provider_response_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyModel:
        def generate_content(self, prompt: str, *, request_options: dict) -> SimpleNamespace:
            return SimpleNamespace(text="```\n```")

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setattr(
        gemini_client,
        "_create_real_model",
        lambda model_name, api_key: EmptyModel(),
    )

    with pytest.raises(gemini_client.GeminiProviderError, match="usable text"):
        gemini_client.keywords_to_sentence(["hello"])


@pytest.mark.parametrize("value", ["", " ", "bad", "nan", "inf", "-inf", "0", "-1", "0.5", "60.1"])
def test_invalid_timeout_fails_before_provider_setup(monkeypatch, value) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", value)
    monkeypatch.setattr(
        gemini_client, "_create_real_model",
        lambda *args: pytest.fail("invalid settings must not initialize the provider"),
    )
    with pytest.raises(gemini_client.GeminiConfigurationError, match="GEMINI_TIMEOUT_SECONDS"):
        gemini_client.keywords_to_sentence(["hello"])


@pytest.mark.parametrize("value, expected", [(None, 15.0), ("1", 1.0), ("60", 60.0)])
def test_timeout_defaults_and_inclusive_bounds(monkeypatch, value, expected) -> None:
    if value is not None:
        monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", value)
    assert gemini_client._settings_from_environment().timeout_seconds == expected


def test_timeout_is_wrapped_without_retry(monkeypatch) -> None:
    calls = []

    class TimedOutModel:
        def generate_content(self, prompt, *, request_options):
            calls.append(request_options)
            raise TimeoutError("private provider details")

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setattr(gemini_client, "_create_real_model", lambda *args: TimedOutModel())
    with pytest.raises(gemini_client.GeminiProviderError) as error:
        gemini_client.keywords_to_sentence(["hello"])
    assert str(error.value) == "Gemini provider request failed"
    assert calls == [{"timeout": 15.0, "retry": None}]
