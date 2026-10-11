"""Google Cloud Text-to-Speech client with an explicit local mock path."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

try:
    from backend.fixtures.mock_audio import MOCK_MP3_BYTES
except ModuleNotFoundError:
    # Supports imports when Uvicorn is started from the backend directory.
    from fixtures.mock_audio import MOCK_MP3_BYTES


AUDIO_CONTENT_TYPE = "audio/mpeg"
DEFAULT_LANGUAGE_CODE = "en-IN"
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


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _settings_from_environment() -> TTSSettings:
    language_code = os.getenv("TTS_LANGUAGE_CODE", DEFAULT_LANGUAGE_CODE).strip()
    voice_name = os.getenv("TTS_VOICE_NAME", "").strip() or None
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
        input_text = texttospeech.SynthesisInput(text=normalized_sentence)

        voice_kwargs: dict[str, Any] = {
            "language_code": settings.language_code,
            "ssml_gender": getattr(
                texttospeech.SsmlVoiceGender, settings.voice_gender
            ),
        }
        if settings.voice_name:
            voice_kwargs["name"] = settings.voice_name

        voice = texttospeech.VoiceSelectionParams(**voice_kwargs)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config,
            timeout=settings.timeout_seconds,
            retry=None,
        )
    except TTSClientError:
        raise
    except Exception as exc:
        raise TTSProviderError(
            "Text-to-Speech provider request failed"
        ) from exc

    return _audio_content(response)
