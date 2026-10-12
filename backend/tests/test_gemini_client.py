"""Deterministic tests for backend module BE-M2."""

from types import SimpleNamespace

import pytest

try:
    from backend.services import gemini_client
except ModuleNotFoundError:
    from services import gemini_client


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
    captured: dict[str, str] = {}

    class FakeModel:
        def generate_content(self, prompt: str) -> SimpleNamespace:
            captured["prompt"] = prompt
            return SimpleNamespace(text="```text\nCan I pay for this medicine?\n```")

    def fake_create_model(model_name: str, api_key: str) -> FakeModel:
        captured["model_name"] = model_name
        captured["api_key"] = api_key
        return FakeModel()

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setenv("GEMINI_MODEL", "configured-test-model")
    monkeypatch.setattr(gemini_client, "_create_real_model", fake_create_model)

    result = gemini_client.keywords_to_sentence(["pay", "medicine", "?"])

    assert result == "Can I pay for this medicine?"
    assert captured["model_name"] == "configured-test-model"
    assert captured["api_key"] == "test-key-not-a-credential"
    assert "pay" in captured["prompt"]
    assert "medicine" in captured["prompt"]


def test_provider_failure_has_a_stable_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "test-key-not-a-credential"

    class FailingModel:
        def generate_content(self, prompt: str) -> None:
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
        def generate_content(self, prompt: str) -> SimpleNamespace:
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


def test_vertex_mode_uses_adc_configuration_and_does_not_require_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def generate_content(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(text="Hello from Vertex AI.")

    def fake_create_model(model_name: str, api_key: str | None, **kwargs: object) -> FakeModel:
        captured.update(kwargs)
        captured["model_name"] = model_name
        captured["api_key"] = api_key
        return FakeModel()

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_USE_VERTEX_AI", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fit-heaven-511209-a9")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_client, "_create_real_model", fake_create_model)

    assert gemini_client.keywords_to_sentence(["hello"]) == "Hello from Vertex AI."
    assert captured == {
        "model_name": gemini_client.DEFAULT_GEMINI_MODEL,
        "api_key": None,
        "use_vertex_ai": True,
        "cloud_project": "fit-heaven-511209-a9",
        "cloud_location": "global",
    }
