"""Google Cloud Text-to-Speech client with an explicit local mock path."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

try:
    from backend.fixtures.mock_audio import (
        MOCK_MP3_BYTES,
        MOCK_MP3_BYTES_EN_IN,
        MOCK_MP3_BYTES_HI_IN,
    )
except ModuleNotFoundError:
    # Supports imports when Uvicorn is started from the backend directory.
    from fixtures.mock_audio import (
        MOCK_MP3_BYTES,
        MOCK_MP3_BYTES_EN_IN,
        MOCK_MP3_BYTES_HI_IN,
    )


AUDIO_CONTENT_TYPE = "audio/mpeg"
DEFAULT_LANGUAGE_CODE = "en-IN"
DEFAULT_HINDI_LANGUAGE_CODE = "hi-IN"
DEFAULT_VOICE_GENDER = "NEUTRAL"
DEFAULT_TIMEOUT_SECONDS = 15.0
MIN_TIMEOUT_SECONDS = 1.0
MAX_TIMEOUT_SECONDS = 60.0
MAX_SENTENCE_LENGTH = 2000
SUPPORTED_VOICE_GENDERS = frozenset({"MALE", "FEMALE", "NEUTRAL"})


class TTSClientError(RuntimeError):
    """Base error for predictable Text-to-Speech client failures."""


class TTSConfigurationError(TTSClientError):
    """Raised when Text-to-Speech configuration is invalid."""


class TTSProviderError(TTSClientError):
    """Raised when the provider cannot return usable MP3 bytes."""


@dataclass(frozen=True)
class TTSSettings:
    """Runtime Text-to-Speech configuration."""

    language_code: str
    voice_name: str | None
    voice_gender: str
    timeout_seconds: float
    mock_external_services: bool
    hindi_language_code: str = DEFAULT_HINDI_LANGUAGE_CODE
    hindi_voice_name: str | None = None


@dataclass(frozen=True)
class SpeechSegment:
    """One labelled MP3 segment ready for the client speaker to play."""

    language: str
    text: str
    audio_content: bytes
    content_type: str = AUDIO_CONTENT_TYPE

    @property
    def language_code(self) -> str:
        return self.language

    @property
    def audio_bytes(self) -> bytes:
        return self.audio_content


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _first_environment_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value.strip()
    return default


def _settings_from_environment() -> TTSSettings:
    language_code = _first_environment_value(
        "TTS_EN_LANGUAGE_CODE", "TTS_ENGLISH_LANGUAGE_CODE", "TTS_LANGUAGE_CODE",
        default=DEFAULT_LANGUAGE_CODE,
    )
    voice_name = _first_environment_value(
        "TTS_EN_VOICE_NAME", "TTS_ENGLISH_VOICE_NAME", "TTS_VOICE_NAME",
    ) or None
    hindi_language_code = _first_environment_value(
        "TTS_HI_LANGUAGE_CODE", "TTS_HINDI_LANGUAGE_CODE",
        default=DEFAULT_HINDI_LANGUAGE_CODE,
    )
    hindi_voice_name = _first_environment_value(
        "TTS_HI_VOICE_NAME", "TTS_HINDI_VOICE_NAME",
    ) or None
    voice_gender = os.getenv("TTS_VOICE_GENDER", DEFAULT_VOICE_GENDER).strip().upper()
    try:
        timeout_seconds = float(
            os.getenv("TTS_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        )
    except (TypeError, ValueError):
        raise TTSConfigurationError(
            "TTS_TIMEOUT_SECONDS must be a finite number from 1 to 60"
        ) from None
    if not math.isfinite(timeout_seconds) or not (
        MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise TTSConfigurationError(
            "TTS_TIMEOUT_SECONDS must be a finite number from 1 to 60"
        )

    if not language_code:
        raise TTSConfigurationError("TTS_LANGUAGE_CODE must not be empty")
    if not hindi_language_code:
        raise TTSConfigurationError("TTS_HI_LANGUAGE_CODE must not be empty")
    if voice_gender not in SUPPORTED_VOICE_GENDERS:
        allowed = ", ".join(sorted(SUPPORTED_VOICE_GENDERS))
        raise TTSConfigurationError(
            f"TTS_VOICE_GENDER must be one of: {allowed}"
        )

    return TTSSettings(
        language_code=language_code,
        voice_name=voice_name,
        voice_gender=voice_gender,
        timeout_seconds=timeout_seconds,
        mock_external_services=_truthy(os.getenv("MOCK_EXTERNAL_SERVICES")),
        hindi_language_code=hindi_language_code,
        hindi_voice_name=hindi_voice_name,
    )


def _validate_sentence(sentence: str) -> str:
    if not isinstance(sentence, str):
        raise ValueError("sentence must be a string")

    normalized = " ".join(sentence.split())
    if not normalized:
        raise ValueError("sentence must not be empty")
    if len(normalized) > MAX_SENTENCE_LENGTH:
        raise ValueError(
            f"sentence must be at most {MAX_SENTENCE_LENGTH} characters"
        )
    return normalized


def _load_texttospeech() -> Any:
    """Load the Google SDK lazily so mock mode has no cloud side effects."""

    try:
        from google.cloud import texttospeech
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TTSConfigurationError(
            "google-cloud-texttospeech is required for the real TTS path"
        ) from exc
    return texttospeech


def _create_real_client(texttospeech: Any) -> Any:
    """Create the ADC-backed client only when the real path is requested."""

    return texttospeech.TextToSpeechClient()


def _audio_content(response: Any) -> bytes:
    try:
        audio_content = response.audio_content
    except Exception as exc:
        raise TTSProviderError("Text-to-Speech returned no usable audio") from exc

    if not isinstance(audio_content, bytes) or not audio_content:
        raise TTSProviderError("Text-to-Speech returned no usable audio")
    return audio_content


def _close_real_client(client: Any) -> None:
    """Close provider transports when the SDK exposes a close method."""

    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # Cleanup must not hide the successful audio or the original provider
            # failure. The client is discarded after this request.
            pass


def _synthesize_with_sdk(
    texttospeech: Any,
    client: Any,
    sentence: str,
    language_code: str,
    voice_name: str | None,
    settings: TTSSettings,
) -> bytes:
    voice_kwargs: dict[str, Any] = {
        "language_code": language_code,
        "ssml_gender": getattr(texttospeech.SsmlVoiceGender, settings.voice_gender),
    }
    if voice_name:
        voice_kwargs["name"] = voice_name

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=sentence),
        voice=texttospeech.VoiceSelectionParams(**voice_kwargs),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        ),
        timeout=settings.timeout_seconds,
        retry=None,
    )
    return _audio_content(response)


def text_to_speech(sentence: str) -> bytes:
    """Synthesize a sentence as MP3 bytes.

    Set ``MOCK_EXTERNAL_SERVICES=1`` for deterministic local/CI output. When
    mock mode is not explicitly enabled, the Google Cloud client uses
    Application Default Credentials and provider failures are wrapped in a
    stable ``TTSProviderError`` for the API layer.
    """

    normalized_sentence = _validate_sentence(sentence)
    settings = _settings_from_environment()

    if settings.mock_external_services:
        return MOCK_MP3_BYTES

    try:
        texttospeech = _load_texttospeech()
        client = _create_real_client(texttospeech)
        try:
            return _synthesize_with_sdk(
                texttospeech,
                client,
                normalized_sentence,
                settings.language_code,
                settings.voice_name,
                settings,
            )
        finally:
            _close_real_client(client)
    except TTSClientError:
        raise
    except Exception as exc:
        raise TTSProviderError(
            "Text-to-Speech provider request failed"
        ) from exc


def text_to_speech_segments(
    english_sentence: str,
    hindi_sentence: str,
) -> tuple[SpeechSegment, SpeechSegment]:
    """Generate English then Hindi MP3 segments for ordered speaker playback.

    The backend returns labelled segments; the mobile speaker should play the
    tuple in order and wait for the first segment to finish before the second.
    No audio device is accessed by this backend client.
    """

    normalized_english = _validate_sentence(english_sentence)
    normalized_hindi = _validate_sentence(hindi_sentence)
    settings = _settings_from_environment()

    if settings.mock_external_services:
        return (
            SpeechSegment(
                settings.language_code,
                normalized_english,
                MOCK_MP3_BYTES_EN_IN,
            ),
            SpeechSegment(
                settings.hindi_language_code,
                normalized_hindi,
                MOCK_MP3_BYTES_HI_IN,
            ),
        )

    try:
        texttospeech = _load_texttospeech()
        client = _create_real_client(texttospeech)
        try:
            english_audio = _synthesize_with_sdk(
                texttospeech,
                client,
                normalized_english,
                settings.language_code,
                settings.voice_name,
                settings,
            )
            hindi_audio = _synthesize_with_sdk(
                texttospeech,
                client,
                normalized_hindi,
                settings.hindi_language_code,
                settings.hindi_voice_name,
                settings,
            )
        finally:
            _close_real_client(client)
    except TTSClientError:
        raise
    except Exception as exc:
        raise TTSProviderError(
            "Text-to-Speech provider request failed"
        ) from exc

    return (
        SpeechSegment(settings.language_code, normalized_english, english_audio),
        SpeechSegment(settings.hindi_language_code, normalized_hindi, hindi_audio),
    )


# Explicit descriptive alias for callers that prefer the operation name.
synthesize_bilingual_speech = text_to_speech_segments
text_to_speech_bilingual = text_to_speech_segments
