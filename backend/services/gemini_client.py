"""Gemini sentence generation client.

The client keeps provider setup lazy so importing the backend, or calling its
health check, never requires a credential or makes a network request. Local
and CI tests must opt into the deterministic mock with
``MOCK_EXTERNAL_SERVICES=1``. The real provider path is used for every other
value and requires ``GEMINI_API_KEY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


# Keep this configurable because model availability changes over time.  This is
# the current model name returned by the provider for this project at runtime.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
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
    mock_external_services: bool
    use_vertex_ai: bool
    cloud_project: str | None
    cloud_location: str


def _truthy(value: str | None) -> bool:
    """Return whether an environment flag explicitly enables a feature."""

    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _settings_from_environment() -> GeminiSettings:
    api_key = os.getenv("GEMINI_API_KEY")
    return GeminiSettings(
        model_name=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        or DEFAULT_GEMINI_MODEL,
        api_key=api_key.strip() if api_key and api_key.strip() else None,
        mock_external_services=_truthy(os.getenv("MOCK_EXTERNAL_SERVICES")),
        use_vertex_ai=_truthy(os.getenv("GEMINI_USE_VERTEX_AI")),
        cloud_project=(os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip() or None,
        cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip() or "global",
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


def _create_real_model(
    model_name: str,
    api_key: str | None,
    *,
    use_vertex_ai: bool = False,
    cloud_project: str | None = None,
    cloud_location: str = "global",
) -> Any:
    """Create a small current-SDK adapter lazily.

    The old ``google-generativeai`` package is deprecated.  Keeping the adapter
    method lets tests and callers retain the existing client boundary while the
    provider SDK can be replaced without changing route code.
    """

    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise GeminiConfigurationError(
            "google-genai is required for the real Gemini path"
        ) from exc

    if use_vertex_ai:
        if not cloud_project:
            raise GeminiConfigurationError(
                "GOOGLE_CLOUD_PROJECT is required for Vertex AI"
            )
        client = genai.Client(
            vertexai=True,
            project=cloud_project,
            location=cloud_location,
        )
    else:
        if not api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY is required for AI Studio")
        client = genai.Client(api_key=api_key)

    class _ModelAdapter:
        def generate_content(self, prompt: str) -> Any:
            return client.models.generate_content(model=model_name, contents=prompt)

    return _ModelAdapter()


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

    if not settings.use_vertex_ai and not settings.api_key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is required when MOCK_EXTERNAL_SERVICES is disabled"
        )

    prompt = _build_prompt(normalized_keywords)
    try:
        if settings.use_vertex_ai:
            model = _create_real_model(
                settings.model_name,
                settings.api_key,
                use_vertex_ai=True,
                cloud_project=settings.cloud_project,
                cloud_location=settings.cloud_location,
            )
        else:
            # Preserve the small two-argument seam used by existing callers and
            # tests for the AI Studio API-key path.
            model = _create_real_model(settings.model_name, settings.api_key)
        response = model.generate_content(prompt)
    except GeminiClientError:
        raise
    except Exception as exc:
        raise GeminiProviderError("Gemini provider request failed") from exc

    return _response_text(response)
