"""Keyword-to-sentence and sentence-to-audio routes for backend module BE-M4."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path as FilePath
from urllib.parse import urlparse

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
    """Resolve a valid absolute base URL for the returned audio link."""

    configured_url = os.getenv("PUBLIC_BASE_URL", "").strip()
    base_url = configured_url or str(request.base_url)
    base_url = base_url.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=500,
            detail="PUBLIC_BASE_URL must be an absolute HTTP(S) URL",
        )
    return base_url


def _audio_file(file_id: str) -> FilePath:
    if not _FILE_ID_PATTERN.fullmatch(file_id):
        raise HTTPException(status_code=404, detail="audio not found")
    return AUDIO_DIR / f"{file_id}.mp3"


@router.post("/process")
def process_keywords(payload: ProcessInput, request: Request) -> dict[str, str]:
    """Turn keywords into one sentence and store its synthesized MP3."""

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

    audio_url = f"{_public_base_url(request)}/audio/{file_id}.mp3"
    return {"sentence": sentence, "audio_url": audio_url}


@router.get("/audio/{file_id}.mp3", response_class=FileResponse)
def audio(file_id: str) -> FileResponse:
    """Serve a previously synthesized local MVP audio file."""

    filename = _audio_file(file_id)
    if not filename.is_file():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(filename, media_type=AUDIO_CONTENT_TYPE)
