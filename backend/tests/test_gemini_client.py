"""Deterministic tests for backend modules BE-M2 and BE-M9."""

import json
import traceback
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


@pytest.mark.parametrize(
    "keywords, expected",
    [
        (
            ["pay", "medicine", "?"],
            {
                "hindi_keywords": ["भुगतान", "दवा", "?"],
                "english_sentence": "Can I pay for this medicine?",
                "hindi_sentence": "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?",
            },
        ),
        (
            ["hello"],
            {
                "hindi_keywords": ["नमस्ते"],
                "english_sentence": "Hello.",
                "hindi_sentence": "नमस्ते।",
            },
        ),
        (
            ["need", "water"],
            {
                "hindi_keywords": ["ज़रूरत", "पानी"],
                "english_sentence": "I need water.",
                "hindi_sentence": "मुझे पानी चाहिए।",
            },
        ),
        (
            ["doctor", "tomorrow", "?"],
            {
                "hindi_keywords": ["डॉक्टर", "कल", "?"],
                "english_sentence": "Can I see the doctor tomorrow?",
                "hindi_sentence": "क्या मैं कल डॉक्टर से मिल सकता हूँ?",
            },
        ),
        (
            ["emergency", "hospital"],
            {
                "hindi_keywords": ["आपातकाल", "अस्पताल"],
                "english_sentence": "It is an emergency; I need to go to the hospital.",
                "hindi_sentence": "यह आपातकाल है; मुझे अस्पताल जाना है।",
            },
        ),
    ],
)
def test_mock_bilingual_text_covers_known_question_and_statement_scenarios(
    monkeypatch: pytest.MonkeyPatch, keywords, expected
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        gemini_client, "_create_real_model",
        lambda *args: pytest.fail("mock mode must not initialize the provider"),
    )

    result = gemini_client.keywords_to_bilingual_text(keywords)
    decision = gemini_client.keywords_to_sentence_decision(keywords)

    assert decision.can_complete_sentence is True
    assert decision.bilingual_text == result
    assert result.hindi_keywords == expected["hindi_keywords"]
    assert result.english_sentence == expected["english_sentence"]
    assert result.hindi_sentence == expected["hindi_sentence"]


def test_mock_bilingual_fallback_is_deterministic_and_bilingual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")

    first = gemini_client.keywords_to_bilingual_text(["help", "food"])
    second = gemini_client.keywords_to_bilingual_text(["help", "food"])

    assert first == second
    assert first.hindi_keywords == ["सहायता", "भोजन"]
    assert any("\u0900" <= character <= "\u097f" for character in first.hindi_sentence)
    assert any(character.isascii() and character.isalpha() for character in first.english_sentence)


@pytest.mark.parametrize(
    "keywords", [["unknown", "topic"], ["unrecognized"], ["hello", "unknown"],
                 ["?"], [".", "!", "।"], ["123"], ["नमस्ते"]],
)
def test_mock_unknown_or_contentless_input_returns_record_again(monkeypatch, keywords):
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    monkeypatch.setattr(
        gemini_client, "_create_real_model",
        lambda *args: pytest.fail("mock decision must not initialize the provider"),
    )
    assert gemini_client.keywords_to_sentence_decision(keywords) == gemini_client.SentenceDecision(
        False, None
    )
    with pytest.raises(gemini_client.GeminiIncompleteSentenceError) as error:
        gemini_client.keywords_to_bilingual_text(keywords)
    assert str(error.value) == "Gemini could not complete a sentence; please record again"


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


def test_real_bilingual_path_returns_structured_text_and_uses_strict_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def generate_content(self, prompt: str, *, request_options: dict) -> SimpleNamespace:
            captured["prompt"] = prompt
            captured["request_options"] = request_options
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "can_complete_sentence": True,
                        "hindi_keywords": ["भुगतान", "दवा", "?"],
                        "english_sentence": "Can I pay for this medicine?",
                        "hindi_sentence": "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?",
                    },
                    ensure_ascii=False,
                )
            )

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setenv("GEMINI_MODEL", "configured-test-model")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "12.5")
    def create_model(model, key):
        captured["model"] = model
        captured["key"] = key
        return FakeModel()

    monkeypatch.setattr(gemini_client, "_create_real_model", create_model)

    decision = gemini_client.keywords_to_sentence_decision(["pay", "medicine", "?"])
    assert decision.can_complete_sentence is True
    result = decision.bilingual_text

    assert result.hindi_keywords == ["भुगतान", "दवा", "?"]
    assert result.english_sentence.startswith("Can I pay")
    assert "Devanagari" in captured["prompt"]
    assert "ONLY strict JSON" in captured["prompt"]
    assert "hindi_keywords, english_sentence, hindi_sentence" in captured["prompt"]
    assert "markdown" in captured["prompt"]
    assert '{"can_complete_sentence": false}' in captured["prompt"]
    assert json.dumps(["pay", "medicine", "?"]) in captured["prompt"]
    assert captured["model"] == "configured-test-model"
    assert captured["key"] == "test-key-not-a-credential"
    assert captured["request_options"] == {"timeout": 12.5, "retry": None}


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "```json\n{\"can_complete_sentence\": false}\n```",
        {"can_complete_sentence": True, "hindi_keywords": ["नमस्ते"], "english_sentence": "Hello."},
        {"can_complete_sentence": True, "hindi_keywords": ["नमस्ते"], "english_sentence": "Hello.", "hindi_sentence": "नमस्ते।", "extra": "nope"},
        {"can_complete_sentence": True, "hindi_keywords": ["नमस्ते"], "english_sentence": "", "hindi_sentence": "नमस्ते।"},
        {"can_complete_sentence": True, "hindi_keywords": ["नमस्ते"], "english_sentence": "Hello.", "hindi_sentence": "Hello."},
        {"can_complete_sentence": True, "hindi_keywords": [3], "english_sentence": "Hello.", "hindi_sentence": "नमस्ते।"},
    ],
)
def test_real_bilingual_path_rejects_invalid_structured_output(
    monkeypatch: pytest.MonkeyPatch, payload
) -> None:
    class FakeModel:
        def generate_content(self, prompt: str, *, request_options: dict) -> SimpleNamespace:
            text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            return SimpleNamespace(text=text)

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setattr(gemini_client, "_create_real_model", lambda *args: FakeModel())

    with pytest.raises(gemini_client.GeminiProviderError, match="no usable bilingual text"):
        gemini_client.keywords_to_bilingual_text(["hello"])


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


@pytest.fixture
def bilingual_response(monkeypatch):
    """Feed raw fake provider output through the public decision API."""
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")

    def invoke(text):
        class FakeModel:
            def generate_content(self, prompt, *, request_options):
                return SimpleNamespace(text=text)

        monkeypatch.setattr(gemini_client, "_create_real_model", lambda *args: FakeModel())
        return gemini_client.keywords_to_sentence_decision(["hello"])

    return invoke


def test_real_false_decision_and_compatibility_error(bilingual_response):
    assert bilingual_response('{"can_complete_sentence":false}') == gemini_client.SentenceDecision(
        False, None
    )
    with pytest.raises(gemini_client.GeminiIncompleteSentenceError) as error:
        gemini_client.keywords_to_bilingual_text(["hello"])
    assert str(error.value) == "Gemini could not complete a sentence; please record again"


@pytest.mark.parametrize("flag", [None, 0, 1, "true", "false", [], {}])
@pytest.mark.parametrize("with_text", [False, True])
def test_decision_requires_strict_boolean(bilingual_response, flag, with_text):
    payload = {"can_complete_sentence": flag}
    if with_text:
        payload.update(
            hindi_keywords=["नमस्ते"], english_sentence="Hello.", hindi_sentence="नमस्ते।"
        )
    with pytest.raises(gemini_client.GeminiProviderError, match="no usable bilingual text"):
        bilingual_response(json.dumps(payload))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"can_complete_sentence": True},
        {"can_complete_sentence": False, "hindi_keywords": []},
        {"can_complete_sentence": False, "english_sentence": "Hello."},
        {"can_complete_sentence": False, "hindi_sentence": None},
        {"can_complete_sentence": False, "reason": "record again"},
        {"can_complete_sentence": False, "usageMetadata": {}},
        {"can_complete_sentence": True, "hindi_keywords": ["नमस्ते"],
         "english_sentence": "Hello.", "hindi_sentence": "नमस्ते।", "usageMetadata": {}},
    ],
)
def test_decision_rejects_incorrect_schema_and_metadata(bilingual_response, payload):
    with pytest.raises(gemini_client.GeminiProviderError, match="no usable bilingual text"):
        bilingual_response(json.dumps(payload))


def test_real_accepted_decision_inclusive_bounds_and_compatibility(bilingual_response):
    payload = {
        "can_complete_sentence": True,
        "hindi_keywords": ["ह" * gemini_client.MAX_HINDI_KEYWORD_LENGTH] * gemini_client.MAX_KEYWORDS,
        "english_sentence": "a" * gemini_client.MAX_SENTENCE_LENGTH,
        "hindi_sentence": "ह" * gemini_client.MAX_SENTENCE_LENGTH,
    }
    raw = json.dumps(payload, ensure_ascii=False)
    raw += " " * (gemini_client.MAX_BILINGUAL_RESPONSE_LENGTH - len(raw))
    decision = bilingual_response(raw)
    expected = gemini_client.BilingualText(
        payload["hindi_keywords"], payload["english_sentence"], payload["hindi_sentence"]
    )
    assert decision == gemini_client.SentenceDecision(True, expected)
    assert gemini_client.keywords_to_bilingual_text(["hello"]) == expected


@pytest.mark.parametrize("keywords", [[" HELP ", " FOOD ", "?"], ["pay", "hello", "need", "water", "doctor", "tomorrow", "emergency", "hospital", "help", "food", "please", "medicine", ".", "!", "।"]])
def test_mock_accepts_known_word_combinations(monkeypatch, keywords):
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    decision = gemini_client.keywords_to_sentence_decision(keywords)
    assert decision.can_complete_sentence is True
    assert decision.bilingual_text == gemini_client.keywords_to_bilingual_text(keywords)


@pytest.mark.parametrize(
    "field, value",
    [
        ("hindi_keywords", []),
        ("hindi_keywords", "नमस्ते"),
        ("hindi_keywords", [None]),
        ("hindi_keywords", [True]),
        ("hindi_keywords", ["  "]),
        ("hindi_keywords", ["hello"]),
        ("hindi_keywords", ["?"]),
        ("hindi_keywords", ["नमस्ते"] * (gemini_client.MAX_KEYWORDS + 1)),
        ("hindi_keywords", ["ह" * (gemini_client.MAX_HINDI_KEYWORD_LENGTH + 1)]),
        ("english_sentence", "नमस्ते।"),
        ("english_sentence", "  "),
        ("english_sentence", 42),
        ("english_sentence", "a" * (gemini_client.MAX_SENTENCE_LENGTH + 1)),
        ("hindi_sentence", "  "),
        ("hindi_sentence", None),
        ("hindi_sentence", "१२३।"),
        ("hindi_sentence", "ह" * (gemini_client.MAX_SENTENCE_LENGTH + 1)),
        ("hindi_sentence", "```नमस्ते।```"),
    ],
)
def test_bilingual_response_rejects_invalid_field_values(bilingual_response, field, value):
    payload = {
        "can_complete_sentence": True,
        "hindi_keywords": ["नमस्ते"],
        "english_sentence": "Hello.",
        "hindi_sentence": "नमस्ते।",
    }
    payload[field] = value
    with pytest.raises(gemini_client.GeminiProviderError) as error:
        bilingual_response(json.dumps(payload, ensure_ascii=False))
    assert str(error.value) == "Gemini returned no usable bilingual text"


@pytest.mark.parametrize(
    "text",
    [
        None, b"{}", "", "  ", "[]", "null", "{} {}",
        '{"can_complete_sentence":true,"hindi_keywords":["नमस्ते"],"english_sentence":"Hi",'
        '"english_sentence":"Hello","hindi_sentence":"नमस्ते।"}',
        '{"can_complete_sentence":true,"hindi_keywords":["नमस्ते"],"english_sentence":NaN,"hindi_sentence":"नमस्ते।"}',
        '{"can_complete_sentence":false,"can_complete_sentence":true}',
        "[" * 1100 + "]" * 1100,
    ],
)
def test_bilingual_response_rejects_unsafe_json(bilingual_response, text):
    with pytest.raises(gemini_client.GeminiProviderError) as error:
        bilingual_response(text)
    assert str(error.value) == "Gemini returned no usable bilingual text"


def test_bilingual_response_rejects_oversized_raw_response(bilingual_response):
    with pytest.raises(gemini_client.GeminiProviderError) as error:
        bilingual_response("x" * (gemini_client.MAX_BILINGUAL_RESPONSE_LENGTH + 1))
    assert str(error.value) == "Gemini returned no usable bilingual text"


@pytest.mark.parametrize("field", ["can_complete_sentence", "hindi_keywords", "english_sentence", "hindi_sentence"])
def test_bilingual_response_requires_every_key(bilingual_response, field):
    payload = {
        "can_complete_sentence": True,
        "hindi_keywords": ["नमस्ते"], "english_sentence": "Hello.",
        "hindi_sentence": "नमस्ते।",
    }
    del payload[field]
    with pytest.raises(gemini_client.GeminiProviderError):
        bilingual_response(json.dumps(payload))


@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError])
def test_bilingual_provider_failure_is_safe_and_not_retried(monkeypatch, failure):
    calls = []

    class FailingModel:
        def generate_content(self, prompt, *, request_options):
            calls.append(request_options)
            raise failure("private provider details and credentials")

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setattr(gemini_client, "_create_real_model", lambda *args: FailingModel())
    with pytest.raises(gemini_client.GeminiProviderError) as error:
        gemini_client.keywords_to_sentence_decision(["hello"])
    assert str(error.value) == "Gemini provider request failed"
    assert "private provider" not in "".join(traceback.format_exception(error.value))
    assert calls == [{"timeout": 15.0, "retry": None}]


def test_bilingual_response_property_failure_is_safe(monkeypatch):
    class BlockedResponse:
        @property
        def text(self):
            raise RuntimeError("private blocked-response details")

    class FakeModel:
        def generate_content(self, prompt, *, request_options):
            return BlockedResponse()

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setattr(gemini_client, "_create_real_model", lambda *args: FakeModel())
    with pytest.raises(gemini_client.GeminiProviderError) as error:
        gemini_client.keywords_to_bilingual_text(["hello"])
    assert str(error.value) == "Gemini returned no usable bilingual text"
    assert "private blocked-response" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("flag", [None, "0"])
def test_bilingual_real_mode_requires_key(monkeypatch, flag):
    monkeypatch.delenv("MOCK_EXTERNAL_SERVICES", raising=False)
    if flag is not None:
        monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", flag)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        gemini_client, "_create_real_model",
        lambda *args: pytest.fail("missing key must fail before provider setup"),
    )
    with pytest.raises(gemini_client.GeminiConfigurationError, match="GEMINI_API_KEY"):
        gemini_client.keywords_to_bilingual_text(["hello"])


@pytest.mark.parametrize(
    "keywords",
    [None, "hello", ("hello",), [], [" "], [1],
     ["x" * (gemini_client.MAX_KEYWORD_LENGTH + 1)],
     ["hello"] * (gemini_client.MAX_KEYWORDS + 1)],
)
@pytest.mark.parametrize("mock_mode", ["0", "1"])
@pytest.mark.parametrize("api", ["keywords_to_bilingual_text", "keywords_to_sentence_decision"])
def test_bilingual_input_bounds_fail_before_provider(monkeypatch, keywords, mock_mode, api):
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", mock_mode)
    monkeypatch.setattr(
        gemini_client, "_create_real_model",
        lambda *args: pytest.fail("invalid input must not initialize provider"),
    )
    with pytest.raises(ValueError):
        getattr(gemini_client, api)(keywords)


def test_bilingual_input_inclusive_bounds_and_normalization(monkeypatch):
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    assert gemini_client.keywords_to_bilingual_text([" hello "]).english_sentence == "Hello."
    result = gemini_client.keywords_to_bilingual_text(
        ["hello"] * gemini_client.MAX_KEYWORDS
    )
    assert len(result.hindi_keywords) == gemini_client.MAX_KEYWORDS
    assert len(result.english_sentence) <= gemini_client.MAX_SENTENCE_LENGTH
    assert len(result.hindi_sentence) <= gemini_client.MAX_SENTENCE_LENGTH
    assert gemini_client.keywords_to_sentence_decision(
        ["x" * gemini_client.MAX_KEYWORD_LENGTH] * gemini_client.MAX_KEYWORDS
    ) == gemini_client.SentenceDecision(False, None)


@pytest.mark.parametrize("value", ["bad", "nan", "inf", "0", "60.1"])
def test_bilingual_timeout_is_validated_before_provider(monkeypatch, value):
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-credential")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", value)
    monkeypatch.setattr(
        gemini_client, "_create_real_model",
        lambda *args: pytest.fail("invalid timeout must not initialize provider"),
    )
    with pytest.raises(gemini_client.GeminiConfigurationError, match="GEMINI_TIMEOUT_SECONDS"):
        gemini_client.keywords_to_bilingual_text(["hello"])
