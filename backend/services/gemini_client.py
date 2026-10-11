"""Gemini sentence generation client.

The client keeps provider setup lazy so importing the backend, or calling its
health check, never requires a credential or makes a network request. Local
and CI tests must opt into the deterministic mock with
``MOCK_EXTERNAL_SERVICES=1``. The real provider path is used for every other
value and requires ``GEMINI_API_KEY``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_TIMEOUT_SECONDS = 15.0
MIN_TIMEOUT_SECONDS = 1.0
MAX_TIMEOUT_SECONDS = 60.0
MAX_KEYWORDS = 20
MAX_KEYWORD_LENGTH = 64


class GeminiClientError(RuntimeError):
    """Base error for predictable Gemini client failures."""


class GeminiConfigurationError(GeminiClientError):
    """Raised when the real provider is not configured safely."""


class GeminiProviderError(GeminiClientError):
    """Raised when Gemini cannot provide a usable sentence."""


@dataclass(frozen=True)
class GeminiSettings:
    """Runtime settings read for each request so tests and deployments can configure them."""

    model_name: str
    api_key: str | None
    timeout_seconds: float
    mock_external_services: bool


def _truthy(value: str | None) -> bool:
    """Return whether an environment flag explicitly enables a feature."""

    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _settings_from_environment() -> GeminiSettings:
    api_key = os.getenv("GEMINI_API_KEY")
    try:
        timeout_seconds = float(
            os.getenv("GEMINI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        )
    except (TypeError, ValueError):
        raise GeminiConfigurationError(
            "GEMINI_TIMEOUT_SECONDS must be a finite number from 1 to 60"
        ) from None
    if not math.isfinite(timeout_seconds) or not (
        MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise GeminiConfigurationError(
            "GEMINI_TIMEOUT_SECONDS must be a finite number from 1 to 60"
        )
    return GeminiSettings(
        model_name=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        or DEFAULT_GEMINI_MODEL,
        api_key=api_key.strip() if api_key and api_key.strip() else None,
        timeout_seconds=timeout_seconds,
        mock_external_services=_truthy(os.getenv("MOCK_EXTERNAL_SERVICES")),
    )


def _validate_keywords(keywords: list[str]) -> list[str]:
    """Validate and normalize the bounded keyword input before prompting Gemini."""

    if not isinstance(keywords, list):
        raise ValueError("keywords must be a non-empty list of strings")
    if not keywords:
        raise ValueError("keywords must not be empty")
    if len(keywords) > MAX_KEYWORDS:
        raise ValueError(f"keywords must contain at most {MAX_KEYWORDS} items")

    normalized: list[str] = []
    for keyword in keywords:
        if not isinstance(keyword, str):
            raise ValueError("every keyword must be a string")
        cleaned = keyword.strip()
        if not cleaned:
            raise ValueError("keywords must not contain blank strings")
        if len(cleaned) > MAX_KEYWORD_LENGTH:
            raise ValueError(
                f"each keyword must be at most {MAX_KEYWORD_LENGTH} characters"
            )
        normalized.append(cleaned)
    return normalized


def _build_prompt(keywords: list[str]) -> str:
    """Build the constrained prompt sent to the provider."""

    keyword_text = ", ".join(keywords)
    return (
        "Turn this keyword sequence into one natural, grammatically correct "
        "spoken sentence. "
        f"Keywords: {keyword_text}. "
        "Output only the sentence, nothing else."
    )


def _mock_sentence(keywords: list[str]) -> str:
    """Return deterministic demo output without calling a paid service."""

    question = "?" in keywords
    words = [keyword for keyword in keywords if keyword not in {"?", "."}]
    phrase = " ".join(words)
    if question:
        return f"Can I {phrase}?"
    return f"{phrase[:1].upper()}{phrase[1:]}."


def _create_real_model(model_name: str, api_key: str) -> Any:
    """Create the SDK model lazily so mock mode has no provider side effects."""

    try:
        import google.generativeai as genai
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise GeminiConfigurationError(
            "google-generativeai is required for the real Gemini path"
        ) from exc

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _response_text(response: Any) -> str:
    """Read and clean the provider response without leaking SDK errors."""

    try:
        text = response.text
    except Exception as exc:  # SDK may raise when a response is blocked/empty.
        raise GeminiProviderError("Gemini returned no usable text") from exc

    if not isinstance(text, str):
        raise GeminiProviderError("Gemini returned no usable text")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        raise GeminiProviderError("Gemini returned no usable text")
    return cleaned


def keywords_to_sentence(keywords: list[str]) -> str:
    """Convert bounded sign keywords into one sentence.

    ``MOCK_EXTERNAL_SERVICES=1`` is the only path that skips the provider. If
    mock mode is not explicitly enabled, a missing key fails fast rather than
    silently using fake output.
    """

    normalized_keywords = _validate_keywords(keywords)
    settings = _settings_from_environment()

    if settings.mock_external_services:
        return _mock_sentence(normalized_keywords)

    if not settings.api_key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is required when MOCK_EXTERNAL_SERVICES is disabled"
        )

    prompt = _build_prompt(normalized_keywords)
    try:
        model = _create_real_model(settings.model_name, settings.api_key)
        response = model.generate_content(
            prompt,
            request_options={"timeout": settings.timeout_seconds, "retry": None},
        )
    except GeminiClientError:
        raise
    except Exception as exc:
        raise GeminiProviderError("Gemini provider request failed") from exc

    return _response_text(response)
