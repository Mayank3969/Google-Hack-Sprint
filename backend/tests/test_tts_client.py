"""Deterministic tests for backend module BE-M3."""

from types import SimpleNamespace

import pytest

try:
    from backend.fixtures.mock_audio import (
        MOCK_MP3_BYTES,
        MOCK_MP3_BYTES_EN_IN,
        MOCK_MP3_BYTES_HI_IN,
        MOCK_MP3_MARKER,
    )
    from backend.services import tts_client
except ModuleNotFoundError:
    from fixtures.mock_audio import (
        MOCK_MP3_BYTES,
        MOCK_MP3_BYTES_EN_IN,
        MOCK_MP3_BYTES_HI_IN,
        MOCK_MP3_MARKER,
    )
    from services import tts_client


@pytest.fixture(autouse=True)
def default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TTS_TIMEOUT_SECONDS", raising=False)


def test_mock_mode_returns_non_empty_recognizable_mp3_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")

    audio = tts_client.text_to_speech("Can I pay for this medicine?")

    assert audio == MOCK_MP3_BYTES
    assert audio.startswith(b"ID3")
    assert MOCK_MP3_MARKER in audio
    assert len(audio) > 0
    assert tts_client.AUDIO_CONTENT_TYPE == "audio/mpeg"


def test_mock_bilingual_segments_are_distinct_and_ordered_for_speaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    monkeypatch.setattr(
        tts_client,
        "_load_texttospeech",
        lambda: pytest.fail("mock TTS must not load Google"),
    )

    segments = tts_client.text_to_speech_segments(
        "Can I pay for this medicine?", "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?"
    )

    assert [segment.language for segment in segments] == ["en-IN", "hi-IN"]
    assert [segment.text for segment in segments] == [
        "Can I pay for this medicine?",
        "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?",
    ]
    assert [segment.audio_content for segment in segments] == [
        MOCK_MP3_BYTES_EN_IN,
        MOCK_MP3_BYTES_HI_IN,
    ]
    assert segments[0].audio_content != segments[1].audio_content
    assert all(segment.audio_content.startswith(b"ID3") for segment in segments)
    assert all(segment.content_type == "audio/mpeg" for segment in segments)
    assert all(segment.audio_bytes == segment.audio_content for segment in segments)
    assert all(segment.language_code == segment.language for segment in segments)


@pytest.mark.parametrize("sentence", ["", "   ", None, "x" * 2001])
def test_sentence_validation_rejects_invalid_input(
    monkeypatch: pytest.MonkeyPatch, sentence: object
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")

    with pytest.raises(ValueError):
        tts_client.text_to_speech(sentence)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "english_sentence, hindi_sentence",
    [("", "नमस्ते।"), ("Hello.", ""), ("   ", "नमस्ते।"),
     ("Hello.", "   "), ("x" * 2001, "नमस्ते।"), ("Hello.", "ह" * 2001)],
)
def test_bilingual_sentence_validation_rejects_invalid_input(
    monkeypatch: pytest.MonkeyPatch, english_sentence: str, hindi_sentence: str
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")

    with pytest.raises(ValueError):
        tts_client.text_to_speech_segments(english_sentence, hindi_sentence)


def test_invalid_voice_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    monkeypatch.setenv("TTS_VOICE_GENDER", "unknown")

    with pytest.raises(tts_client.TTSConfigurationError, match="TTS_VOICE_GENDER"):
        tts_client.text_to_speech("hello")


def test_real_path_uses_configured_voice_and_returns_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected_audio = b"real-mp3-test-bytes"

    class FakeTextToSpeech:
        class SsmlVoiceGender:
            NEUTRAL = "neutral"

        class AudioEncoding:
            MP3 = "mp3"

        @staticmethod
        def SynthesisInput(**kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

        @staticmethod
        def VoiceSelectionParams(**kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

        @staticmethod
        def AudioConfig(**kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

    class FakeClient:
        def synthesize_speech(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(audio_content=expected_audio)

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("TTS_LANGUAGE_CODE", "en-IN")
    monkeypatch.setenv("TTS_VOICE_NAME", "en-IN-Neural2-A")
    monkeypatch.setenv("TTS_VOICE_GENDER", "NEUTRAL")
    monkeypatch.setenv("TTS_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setattr(tts_client, "_load_texttospeech", lambda: FakeTextToSpeech)
    monkeypatch.setattr(
        tts_client,
        "_create_real_client",
        lambda texttospeech: FakeClient(),
    )

    audio = tts_client.text_to_speech("  hello   from SignBridge  ")

    assert audio == expected_audio
    assert captured["input"].text == "hello from SignBridge"
    assert captured["voice"].language_code == "en-IN"
    assert captured["voice"].name == "en-IN-Neural2-A"
    assert captured["voice"].ssml_gender == "neutral"
    assert captured["audio_config"].audio_encoding == "mp3"
    assert captured["timeout"] == 12.5
    assert captured["retry"] is None


def test_real_bilingual_path_uses_both_languages_and_closes_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    state = {"closed": False}

    class FakeTextToSpeech:
        class SsmlVoiceGender:
            NEUTRAL = "neutral"

        class AudioEncoding:
            MP3 = "mp3"

        @staticmethod
        def SynthesisInput(**kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

        @staticmethod
        def VoiceSelectionParams(**kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

        @staticmethod
        def AudioConfig(**kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

    class FakeClient:
        def synthesize_speech(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            language = kwargs["voice"].language_code
            return SimpleNamespace(audio_content=(b"english-real" if language == "en-IN" else b"hindi-real"))

        def close(self) -> None:
            state["closed"] = True

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("TTS_EN_LANGUAGE_CODE", "en-IN")
    monkeypatch.setenv("TTS_HI_LANGUAGE_CODE", "hi-IN")
    monkeypatch.setenv("TTS_EN_VOICE_NAME", "en-IN-Neural2-A")
    monkeypatch.setenv("TTS_HI_VOICE_NAME", "hi-IN-Neural2-A")
    monkeypatch.setenv("TTS_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setattr(tts_client, "_load_texttospeech", lambda: FakeTextToSpeech)
    monkeypatch.setattr(tts_client, "_create_real_client", lambda sdk: FakeClient())

    segments = tts_client.text_to_speech_segments(
        "  Hello from SignBridge. ", "  नमस्ते साइनब्रिज से। "
    )

    assert [(segment.language, segment.audio_content) for segment in segments] == [
        ("en-IN", b"english-real"),
        ("hi-IN", b"hindi-real"),
    ]
    assert [call["voice"].language_code for call in calls] == ["en-IN", "hi-IN"]
    assert [call["voice"].name for call in calls] == [
        "en-IN-Neural2-A", "hi-IN-Neural2-A"
    ]
    assert [call["input"].text for call in calls] == [
        "Hello from SignBridge.", "नमस्ते साइनब्रिज से।"
    ]
    assert all(call["audio_config"].audio_encoding == "mp3" for call in calls)
    assert all(call["timeout"] == 12.5 and call["retry"] is None for call in calls)
    assert state["closed"] is True


def test_real_bilingual_failure_does_not_return_partial_audio_and_closes_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    state = {"closed": False}

    class FakeTextToSpeech(FakeSdk):
        pass

    class FailingClient:
        def synthesize_speech(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(audio_content=b"english-real")
            raise RuntimeError("provider details must not escape")

        def close(self):
            state["closed"] = True

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(tts_client, "_load_texttospeech", lambda: FakeTextToSpeech)
    monkeypatch.setattr(tts_client, "_create_real_client", lambda sdk: FailingClient())

    with pytest.raises(tts_client.TTSProviderError) as error:
        tts_client.text_to_speech_segments("Hello.", "नमस्ते।")

    assert str(error.value) == "Text-to-Speech provider request failed"
    assert "provider details" not in str(error.value)
    assert len(calls) == 2
    assert state["closed"] is True


def test_real_bilingual_empty_audio_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyClient:
        def synthesize_speech(self, **kwargs):
            return SimpleNamespace(audio_content=b"")

        def close(self):
            pass

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(tts_client, "_load_texttospeech", lambda: FakeSdk)
    monkeypatch.setattr(tts_client, "_create_real_client", lambda sdk: EmptyClient())

    with pytest.raises(tts_client.TTSProviderError, match="no usable audio"):
        tts_client.text_to_speech_segments("Hello.", "नमस्ते।")


def test_provider_failure_has_a_stable_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        def synthesize_speech(self, **kwargs: object) -> None:
            raise RuntimeError("provider details should not escape")

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(tts_client, "_load_texttospeech", lambda: FakeSdk)
    monkeypatch.setattr(
        tts_client,
        "_create_real_client",
        lambda texttospeech: FailingClient(),
    )

    with pytest.raises(tts_client.TTSProviderError) as error:
        tts_client.text_to_speech("hello")

    assert str(error.value) == "Text-to-Speech provider request failed"


class FakeSdk:
    """Minimal SDK surface for the provider failure test."""

    class SsmlVoiceGender:
        NEUTRAL = "neutral"

    class AudioEncoding:
        MP3 = "mp3"

    @staticmethod
    def SynthesisInput(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    @staticmethod
    def VoiceSelectionParams(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    @staticmethod
    def AudioConfig(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)


@pytest.mark.parametrize("value", ["", " ", "bad", "nan", "inf", "-inf", "0", "-1", "0.5", "60.1"])
def test_invalid_timeout_fails_before_provider_setup(monkeypatch, value) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("TTS_TIMEOUT_SECONDS", value)
    monkeypatch.setattr(
        tts_client, "_load_texttospeech",
        lambda: pytest.fail("invalid settings must not initialize the provider"),
    )
    with pytest.raises(tts_client.TTSConfigurationError, match="TTS_TIMEOUT_SECONDS"):
        tts_client.text_to_speech("hello")


@pytest.mark.parametrize("value, expected", [(None, 15.0), ("1", 1.0), ("60", 60.0)])
def test_timeout_defaults_and_inclusive_bounds(monkeypatch, value, expected) -> None:
    if value is not None:
        monkeypatch.setenv("TTS_TIMEOUT_SECONDS", value)
    assert tts_client._settings_from_environment().timeout_seconds == expected


def test_timeout_is_wrapped_without_retry(monkeypatch) -> None:
    calls = []

    class TimedOutClient:
        def synthesize_speech(self, **kwargs):
            calls.append(kwargs)
            raise TimeoutError("private provider details")

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(tts_client, "_load_texttospeech", lambda: FakeSdk)
    monkeypatch.setattr(tts_client, "_create_real_client", lambda sdk: TimedOutClient())
    with pytest.raises(tts_client.TTSProviderError) as error:
        tts_client.text_to_speech("hello")
    assert str(error.value) == "Text-to-Speech provider request failed"
    assert len(calls) == 1
    assert calls[0]["timeout"] == 15.0
    assert calls[0]["retry"] is None
