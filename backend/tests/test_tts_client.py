"""Deterministic tests for backend module BE-M3."""

from types import SimpleNamespace

import pytest

try:
    from backend.fixtures.mock_audio import MOCK_MP3_BYTES, MOCK_MP3_MARKER
    from backend.services import tts_client
except ModuleNotFoundError:
    from fixtures.mock_audio import MOCK_MP3_BYTES, MOCK_MP3_MARKER
    from services import tts_client


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


@pytest.mark.parametrize("sentence", ["", "   ", None, "x" * 2001])
def test_sentence_validation_rejects_invalid_input(
    monkeypatch: pytest.MonkeyPatch, sentence: object
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")

    with pytest.raises(ValueError):
        tts_client.text_to_speech(sentence)  # type: ignore[arg-type]


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
