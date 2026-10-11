"""Deterministic tests for backend module BE-M5."""

from io import BytesIO
from types import SimpleNamespace
import wave

import pytest

try:
    from backend.services.audio_validation import (
        AudioValidationError,
        MAX_AUDIO_DURATION_SECONDS,
        MAX_AUDIO_BYTES,
        validate_wav,
    )
    from backend.services import stt_client
except ModuleNotFoundError:
    from services.audio_validation import (
        AudioValidationError,
        MAX_AUDIO_DURATION_SECONDS,
        MAX_AUDIO_BYTES,
        validate_wav,
    )
    from services import stt_client


def make_wav(
    *,
    frames: bytes,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return buffer.getvalue()


def test_validate_wav_returns_mobile_compatible_metadata() -> None:
    pcm = b"\x01\x00" * 160
    audio = make_wav(frames=pcm)

    validated = validate_wav(audio)

    assert validated.sample_rate_hz == 16_000
    assert validated.channels == 1
    assert validated.sample_width_bytes == 2
    assert validated.frame_count == 160
    assert validated.duration_seconds == pytest.approx(0.01)
    assert validated.pcm_frames == pcm


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 8_000},
        {"channels": 2},
        {"sample_width": 1},
    ],
)
def test_validate_wav_rejects_unsupported_audio_format(kwargs: dict[str, int]) -> None:
    audio = make_wav(frames=b"\x00\x00" * 160, **kwargs)

    with pytest.raises(AudioValidationError):
        validate_wav(audio)


def test_validate_wav_rejects_empty_malformed_truncated_and_oversized_audio() -> None:
    valid_audio = make_wav(frames=b"\x00\x00" * 160)

    for invalid_audio in (
        b"",
        b"not-a-wav",
        valid_audio[:-1],
        b"x" * (MAX_AUDIO_BYTES + 1),
    ):
        with pytest.raises(AudioValidationError):
            validate_wav(invalid_audio)


def test_validate_wav_rejects_audio_longer_than_thirty_seconds() -> None:
    frames = b"\x00\x00" * (16_000 * MAX_AUDIO_DURATION_SECONDS + 1)
    audio = make_wav(frames=frames)

    with pytest.raises(AudioValidationError, match="duration"):
        validate_wav(audio)


def test_mock_mode_returns_transcript_for_non_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    audio = make_wav(frames=b"\x01\x00" * 160)

    assert stt_client.speech_to_text(audio) == stt_client.MOCK_TRANSCRIPT


def test_mock_mode_returns_empty_text_for_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    audio = make_wav(frames=b"\x00\x00" * 160)

    assert stt_client.speech_to_text(audio) == ""


def test_invalid_audio_is_rejected_before_mock_or_provider_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")

    with pytest.raises(AudioValidationError):
        stt_client.speech_to_text(b"not-a-wav")


def test_invalid_timeout_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    monkeypatch.setenv("STT_TIMEOUT_SECONDS", "not-a-number")

    with pytest.raises(stt_client.STTConfigurationError, match="STT_TIMEOUT_SECONDS"):
        stt_client.speech_to_text(make_wav(frames=b"\x00\x00" * 160))


def test_real_path_uses_speech_config_and_closes_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    audio = make_wav(frames=b"\x01\x00" * 160)

    class FakeSpeech:
        class RecognitionConfig:
            class AudioEncoding:
                LINEAR16 = "linear16"

            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        class RecognitionAudio:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

    class FakeTransport:
        def close(self) -> None:
            captured["closed"] = True

    class FakeClient:
        transport = FakeTransport()

        def recognize(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        alternatives=[SimpleNamespace(transcript="  yes  ")]
                    ),
                    SimpleNamespace(
                        alternatives=[SimpleNamespace(transcript="that's fine")]
                    ),
                ]
            )

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("STT_LANGUAGE_CODE", "en-IN")
    monkeypatch.setenv("STT_TIMEOUT_SECONDS", "12")
    monkeypatch.setattr(stt_client, "_load_speech", lambda: FakeSpeech)
    monkeypatch.setattr(stt_client, "_create_real_client", lambda speech: FakeClient())

    result = stt_client.speech_to_text(audio)

    config = captured["config"]
    request_audio = captured["audio"]
    assert result == "yes that's fine"
    assert config.encoding == "linear16"
    assert config.sample_rate_hertz == 16_000
    assert config.audio_channel_count == 1
    assert config.language_code == "en-IN"
    assert request_audio.content == audio
    assert captured["timeout"] == 12.0
    assert captured["retry"] is None
    assert captured["closed"] is True


def test_provider_failure_is_wrapped_and_transport_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = make_wav(frames=b"\x01\x00" * 160)
    closed = {"value": False}

    class FakeSpeech:
        class RecognitionConfig:
            class AudioEncoding:
                LINEAR16 = "linear16"

        class RecognitionAudio:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

    class FakeTransport:
        def close(self) -> None:
            closed["value"] = True

    class FailingClient:
        transport = FakeTransport()

        def recognize(self, **kwargs: object) -> None:
            raise RuntimeError("provider details should not escape")

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(stt_client, "_load_speech", lambda: FakeSpeech)
    monkeypatch.setattr(stt_client, "_create_real_client", lambda speech: FailingClient())

    with pytest.raises(stt_client.STTProviderError, match="provider request failed"):
        stt_client.speech_to_text(audio)

    assert closed["value"] is True
