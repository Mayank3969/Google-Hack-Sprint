"""Google Cloud Speech-to-Text client with an explicit local mock path."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from .audio_validation import AudioValidationError, validate_wav


DEFAULT_LANGUAGE_CODE = "en-IN"
DEFAULT_TIMEOUT_SECONDS = 15.0
MIN_TIMEOUT_SECONDS = 1.0
MAX_TIMEOUT_SECONDS = 60.0
MOCK_TRANSCRIPT = "[MOCK] Yes, that is fine."


class STTClientError(RuntimeError):
    """Base error for predictable Speech-to-Text client failures."""


class STTConfigurationError(STTClientError):
    """Raised when Speech-to-Text configuration or credentials are unavailable."""


class STTProviderError(STTClientError):
    """Raised when Speech-to-Text cannot return a usable response."""


@dataclass(frozen=True)
class STTSettings:
    """Runtime Speech-to-Text configuration."""

    language_code: str
    timeout_seconds: float
    mock_external_services: bool


def _truthy(value: str | None) -> bool:
    """Return whether an environment flag explicitly enables mock mode."""

    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _settings_from_environment() -> STTSettings:
    """Read and validate request settings without loading a provider SDK."""

    language_code = os.getenv("STT_LANGUAGE_CODE", DEFAULT_LANGUAGE_CODE).strip()
    timeout_text = os.getenv("STT_TIMEOUT_SECONDS", str(int(DEFAULT_TIMEOUT_SECONDS)))

    if not language_code:
        raise STTConfigurationError("STT_LANGUAGE_CODE must not be empty")

    if not timeout_text or not timeout_text.strip():
        raise STTConfigurationError(
            "STT_TIMEOUT_SECONDS must be a finite number from 1 to 60"
        )

    try:
        timeout_seconds = float(timeout_text.strip())
    except (TypeError, ValueError):
        raise STTConfigurationError(
            "STT_TIMEOUT_SECONDS must be a finite number from 1 to 60"
        ) from None

    if not math.isfinite(timeout_seconds) or not (
        MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise STTConfigurationError(
            "STT_TIMEOUT_SECONDS must be a finite number from 1 to 60"
        )

    return STTSettings(
        language_code=language_code,
        timeout_seconds=timeout_seconds,
        mock_external_services=_truthy(os.getenv("MOCK_EXTERNAL_SERVICES")),
    )


def _load_speech() -> Any:
    """Load the Google Speech SDK lazily for the real provider path."""

    try:
        from google.cloud import speech
    except ImportError:
        raise STTConfigurationError(
            "google-cloud-speech is required for the real STT path"
        ) from None
    return speech


def _create_real_client(speech: Any) -> Any:
    """Create an ADC-backed Speech client only when the real path is requested."""

    return speech.SpeechClient()


def _is_missing_adc_error(error: BaseException) -> bool:
    """Recognize missing Application Default Credentials without exposing details."""

    if error.__class__.__name__ in {
        "DefaultCredentialsError",
        "AnonymousCredentialsError",
    }:
        return True

    try:
        message = str(error).lower()
    except Exception:
        return False

    return (
        "application default credential" in message
        or "default credentials" in message
        or "could not automatically determine credentials" in message
        or (
            "adc" in message
            and any(
                marker in message for marker in ("missing", "unavailable", "required")
            )
        )
    )


def _mock_result(validated_wav: Any) -> str:
    """Return deterministic demo output; this path does not perform recognition."""

    try:
        pcm_frames = validated_wav.pcm_frames
    except Exception:
        raise STTProviderError("Speech-to-Text returned unusable audio data") from None

    if not isinstance(pcm_frames, bytes):
        raise STTProviderError("Speech-to-Text returned unusable audio data")
    return MOCK_TRANSCRIPT if any(pcm_frames) else ""


def _transcript_from_response(response: Any) -> str:
    """Extract and normalize the best alternative from every recognition result."""

    try:
        results = response.results
    except Exception:
        raise STTProviderError("Speech-to-Text returned an invalid response") from None

    if results is None:
        return ""
    if isinstance(results, (str, bytes, bytearray)):
        raise STTProviderError("Speech-to-Text returned an invalid response") from None

    transcripts: list[str] = []
    try:
        result_iterator = iter(results)
        for result in result_iterator:
            if result is None:
                raise STTProviderError("Speech-to-Text returned an invalid response")

            try:
                alternatives = result.alternatives
            except Exception:
                raise STTProviderError(
                    "Speech-to-Text returned an invalid response"
                ) from None

            if alternatives is None:
                continue
            if isinstance(alternatives, (str, bytes, bytearray)):
                raise STTProviderError(
                    "Speech-to-Text returned an invalid response"
                ) from None

            alternative_iterator = iter(alternatives)
            try:
                best_alternative = next(alternative_iterator)
            except StopIteration:
                continue

            if best_alternative is None:
                raise STTProviderError("Speech-to-Text returned an invalid response")
            try:
                transcript = best_alternative.transcript
            except Exception:
                raise STTProviderError(
                    "Speech-to-Text returned an invalid response"
                ) from None
            if not isinstance(transcript, str):
                raise STTProviderError(
                    "Speech-to-Text returned an invalid transcript"
                ) from None

            normalized = " ".join(transcript.split())
            if normalized:
                transcripts.append(normalized)
    except STTClientError:
        raise
    except Exception:
        raise STTProviderError("Speech-to-Text returned an invalid response") from None

    return " ".join(transcripts)


def _close_owned_client(client: Any) -> None:
    """Close an owned client transport without changing the request outcome."""

    try:
        transport = getattr(client, "transport", None)
        close_transport = getattr(transport, "close", None)
        if callable(close_transport):
            close_transport()
            return
    except Exception:
        pass

    try:
        close_client = getattr(client, "close", None)
        if callable(close_client):
            close_client()
    except Exception:
        pass


def _linear16_encoding(speech: Any) -> Any:
    """Get the v1 LINEAR16 enum while keeping SDK-shape failures safe."""

    try:
        return speech.RecognitionConfig.AudioEncoding.LINEAR16
    except Exception:
        try:
            return speech.RecognitionConfig.LINEAR16
        except Exception:
            raise STTProviderError(
                "Speech-to-Text SDK returned an unusable configuration type"
            ) from None


def speech_to_text(audio_bytes: bytes) -> str:
    """Recognize validated 16 kHz mono 16-bit PCM WAV bytes.

    ``MOCK_EXTERNAL_SERVICES`` must explicitly be one of ``1``, ``true``,
    ``yes``, or ``on`` to use deterministic local output. Mock output is only
    a development placeholder and is not speech recognition.
    """

    validated_wav = validate_wav(audio_bytes)
    settings = _settings_from_environment()

    if settings.mock_external_services:
        return _mock_result(validated_wav)

    client: Any | None = None
    try:
        try:
            speech = _load_speech()
            client = _create_real_client(speech)
        except STTClientError:
            raise
        except Exception as exc:
            if isinstance(exc, ImportError):
                raise STTConfigurationError(
                    "google-cloud-speech is required for the real STT path"
                ) from None
            if _is_missing_adc_error(exc):
                raise STTConfigurationError(
                    "Google Speech-to-Text Application Default Credentials are unavailable"
                ) from None
            raise STTProviderError("Speech-to-Text client setup failed") from None

        try:
            config = speech.RecognitionConfig(
                encoding=_linear16_encoding(speech),
                sample_rate_hertz=16000,
                audio_channel_count=1,
                language_code=settings.language_code,
            )
            audio = speech.RecognitionAudio(content=audio_bytes)
            response = client.recognize(
                config=config,
                audio=audio,
                timeout=settings.timeout_seconds,
                retry=None,
            )
        except STTClientError:
            raise
        except Exception:
            raise STTProviderError("Speech-to-Text provider request failed") from None

        return _transcript_from_response(response)
    finally:
        if client is not None:
            _close_owned_client(client)


__all__ = [
    "AudioValidationError",
    "MOCK_TRANSCRIPT",
    "STTClientError",
    "STTConfigurationError",
    "STTProviderError",
    "_load_speech",
    "_create_real_client",
    "speech_to_text",
]
