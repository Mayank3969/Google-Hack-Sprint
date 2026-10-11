"""Keyword-to-sentence and sentence-to-audio routes for backend module BE-M4."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from ipaddress import IPv6Address
from pathlib import Path as FilePath
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StrictStr, field_validator

if __package__ and __package__.startswith("backend."):
    from ..services.gemini_client import (
        MAX_KEYWORD_LENGTH,
        MAX_KEYWORDS,
        GeminiClientError,
        keywords_to_sentence,
    )
    from ..services.tts_client import (
        AUDIO_CONTENT_TYPE,
        TTSClientError,
        text_to_speech,
    )
else:  # Supports imports when Uvicorn is started from the backend directory.
    from services.gemini_client import (
        MAX_KEYWORD_LENGTH,
        MAX_KEYWORDS,
        GeminiClientError,
        keywords_to_sentence,
    )
    from services.tts_client import AUDIO_CONTENT_TYPE, TTSClientError, text_to_speech


router = APIRouter(tags=["process"])

_FILE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
AUDIO_DIR = FilePath(
    os.getenv(
        "AUDIO_DIR",
        str(FilePath(tempfile.gettempdir()) / "signbridge-audio"),
    )
)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


class ProcessInput(BaseModel):
    """Validated keyword input for the mobile-facing process endpoint."""

    keywords: list[StrictStr] = Field(
        min_length=1,
        max_length=MAX_KEYWORDS,
        description="Ordered sign keywords, including '?' when the phrase is a question.",
    )

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, keywords: list[str]) -> list[str]:
        normalized: list[str] = []
        for keyword in keywords:
            cleaned = keyword.strip()
            if not cleaned:
                raise ValueError("keywords must not contain blank strings")
            if len(cleaned) > MAX_KEYWORD_LENGTH:
                raise ValueError(
                    f"each keyword must be at most {MAX_KEYWORD_LENGTH} characters"
                )
            normalized.append(cleaned)
        return normalized


def _public_base_url(request: Request) -> str:
    """Validate configuration before paid work; deployed links require HTTPS."""

    https_flag = os.getenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "0").strip().lower()
    if https_flag not in {"0", "false", "no", "off", "1", "true", "yes", "on"}:
        raise HTTPException(
            status_code=500,
            detail="REQUIRE_HTTPS_PUBLIC_BASE_URL must be a boolean flag",
        )
    require_https = bool(os.getenv("K_SERVICE")) or https_flag in {
        "1", "true", "yes", "on"
    }
    configured_url = os.getenv("PUBLIC_BASE_URL", "")
    if require_https and not configured_url.strip():
        raise HTTPException(
            status_code=500,
            detail="PUBLIC_BASE_URL must be configured in deployed environments",
        )
    base_url = (configured_url or str(request.base_url)).rstrip("/")
    try:
        # urlsplit silently removes some control characters, so reject them first.
        if any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in base_url
        ) or any(c in base_url for c in "\\?#"):
            raise ValueError("invalid URL characters")
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        port = parsed.port  # Access validates malformed/out-of-range port values.
        if (
            parsed.scheme not in {"http", "https"}
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.netloc.endswith(":")
            or port == 0
        ):
            raise ValueError("invalid HTTP URL")
        if ":" in hostname:
            if not re.fullmatch(r"\[[0-9a-fA-F:.]+\](?::[0-9]+)?", parsed.netloc):
                raise ValueError("invalid IPv6 authority")
            IPv6Address(hostname)
        else:
            ascii_hostname = hostname.encode("idna").decode("ascii").removesuffix(".")
            if len(ascii_hostname) > 253 or any(
                not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label)
                for label in ascii_hostname.split(".")
            ):
                raise ValueError("invalid hostname")
    except (ValueError, UnicodeError):
        raise HTTPException(
            status_code=500,
            detail="PUBLIC_BASE_URL must be an absolute HTTP(S) URL without credentials, query, or fragment",
        ) from None
    if require_https and parsed.scheme != "https":
        raise HTTPException(
            status_code=500,
            detail="PUBLIC_BASE_URL must use HTTPS in deployed environments",
        )
    return base_url


def _audio_file(file_id: str) -> FilePath:
    if not _FILE_ID_PATTERN.fullmatch(file_id):
        raise HTTPException(status_code=404, detail="audio not found")
    return AUDIO_DIR / f"{file_id}.mp3"


@router.post("/process")
def process_keywords(payload: ProcessInput, request: Request) -> dict[str, str]:
    """Turn keywords into one sentence and store its synthesized MP3."""

    public_base_url = _public_base_url(request)

    try:
        sentence = keywords_to_sentence(payload.keywords)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GeminiClientError as exc:
        raise HTTPException(
            status_code=502,
            detail="sentence generation failed",
        ) from exc

    try:
        audio_bytes = text_to_speech(sentence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TTSClientError as exc:
        raise HTTPException(
            status_code=502,
            detail="speech synthesis failed",
        ) from exc

    file_id = uuid.uuid4().hex
    filename = AUDIO_DIR / f"{file_id}.mp3"
    try:
        filename.write_bytes(audio_bytes)
    except (OSError, TypeError) as exc:
        raise HTTPException(status_code=500, detail="audio storage failed") from exc

    audio_url = f"{public_base_url}/audio/{file_id}.mp3"
    return {"sentence": sentence, "audio_url": audio_url}


@router.get("/audio/{file_id}.mp3", response_class=FileResponse)
def audio(file_id: str) -> FileResponse:
    """Serve a previously synthesized local MVP audio file."""

    filename = _audio_file(file_id)
    if not filename.is_file():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(filename, media_type=AUDIO_CONTENT_TYPE)
