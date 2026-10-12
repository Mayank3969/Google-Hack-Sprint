"""Keyword-to-bilingual-text-and-audio routes for backend module BE-M11."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
import math
from ipaddress import IPv6Address
from pathlib import Path as FilePath
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator, model_validator
from typing import Literal

if __package__ and __package__.startswith("backend."):
    from ..services.gemini_client import (
        MAX_KEYWORD_LENGTH,
        MAX_KEYWORDS,
        GeminiClientError,
        keywords_to_sentence_decision,
    )
    from ..services.tts_client import (
        AUDIO_CONTENT_TYPE,
        TTSClientError,
        text_to_speech_segments,
    )
else:  # Supports imports when Uvicorn is started from the backend directory.
    from services.gemini_client import (
        MAX_KEYWORD_LENGTH,
        MAX_KEYWORDS,
        GeminiClientError,
        keywords_to_sentence_decision,
    )
    from services.tts_client import (
        AUDIO_CONTENT_TYPE,
        TTSClientError,
        text_to_speech_segments,
    )


router = APIRouter(tags=["process"])

_FILE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MAX_CAPTURE_RESULTS = 10
MAX_TOP_K_RESULTS = 20
DEFAULT_RECOGNITION_THRESHOLD = 0.5
RECOGNITION_THRESHOLD_ENV = "SIGNBRIDGE_RECOGNITION_THRESHOLD"


class RecognitionConfigurationError(ValueError):
    """Raised when the server-side ML threshold configuration is invalid."""
AUDIO_DIR = FilePath(
    os.getenv(
        "AUDIO_DIR",
        str(FilePath(tempfile.gettempdir()) / "signbridge-audio"),
    )
)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


class TopKResult(BaseModel):
    """Optional ranked ML candidate included for input-contract validation."""

    label: StrictStr = Field(min_length=1, max_length=MAX_KEYWORD_LENGTH)
    confidence: float = Field(ge=0.0, le=1.0)


class RecognitionResult(BaseModel):
    """One segmented ML recognition result."""

    segment_index: StrictInt = Field(ge=0)
    status: Literal["recognized", "uncertain"]
    keyword: StrictStr | None = Field(default=None, max_length=MAX_KEYWORD_LENGTH)
    confidence: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    top_k: list[TopKResult] | None = Field(default=None, max_length=MAX_TOP_K_RESULTS)


class ProcessInput(BaseModel):
    """Validate legacy keywords or the segmented ML capture-results payload."""

    keywords: list[StrictStr] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_KEYWORDS,
        description="Legacy ordered sign keywords.",
    )
    capture_id: StrictStr | None = Field(default=None, min_length=1, max_length=128)
    results: list[RecognitionResult] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_CAPTURE_RESULTS,
        description="Ordered or indexed segmented ML recognition results.",
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

    @field_validator("capture_id")
    @classmethod
    def validate_capture_id(cls, capture_id: str | None) -> str | None:
        if capture_id is not None and not capture_id.strip():
            raise ValueError("capture_id must not be blank")
        return capture_id.strip() if capture_id is not None else None

    @model_validator(mode="after")
    def validate_input_shape(self) -> "ProcessInput":
        has_legacy = self.keywords is not None
        has_capture = self.capture_id is not None or self.results is not None
        if has_legacy and has_capture:
            raise ValueError("provide either keywords or capture_id/results, not both")
        if not has_legacy and not has_capture:
            raise ValueError("provide keywords or capture_id/results")
        if has_legacy:
            return self
        if self.capture_id is None or self.results is None:
            raise ValueError("capture_id and results are required together")
        if len({result.segment_index for result in self.results}) != len(self.results):
            raise ValueError("results must contain unique segment_index values")
        if not any(
            result.status == "recognized"
            and (
                (result.keyword and result.keyword.strip())
                or any(candidate.label.strip() for candidate in (result.top_k or []))
            )
            for result in self.results
        ):
            raise ValueError("results must contain at least one recognized candidate")
        return self

    def ordered_keywords(self) -> list[str]:
        """Return thresholded, stable, duplicate-free ML words in segment order."""

        if self.keywords is not None:
            return self.keywords
        assert self.results is not None
        threshold = _recognition_threshold()
        selected: list[str] = []
        seen: set[str] = set()
        for result in sorted(self.results, key=lambda result: result.segment_index):
            if (
                result.status != "recognized"
                or result.confidence < threshold
                or result.coverage < threshold
            ):
                continue
            candidates = result.top_k
            if candidates is None:
                candidates = (
                    [TopKResult(label=result.keyword, confidence=result.confidence)]
                    if result.keyword and result.keyword.strip()
                    else []
                )
            for candidate in candidates:
                label = candidate.label.strip()
                if candidate.confidence < threshold or not label:
                    continue
                dedupe_key = label.casefold()
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                selected.append(label)
        if len(selected) > MAX_KEYWORDS:
            raise ValueError(
                f"recognized keywords must contain at most {MAX_KEYWORDS} items "
                "after threshold filtering"
            )
        return selected


def _recognition_threshold() -> float:
    """Return the inclusive confidence/coverage threshold for ML candidates."""

    raw = os.getenv(
        RECOGNITION_THRESHOLD_ENV, str(DEFAULT_RECOGNITION_THRESHOLD)
    ).strip()
    try:
        threshold = float(raw)
    except (TypeError, ValueError):
        raise RecognitionConfigurationError(
            f"{RECOGNITION_THRESHOLD_ENV} must be a finite number from 0 to 1"
        ) from None
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise RecognitionConfigurationError(
            f"{RECOGNITION_THRESHOLD_ENV} must be a finite number from 0 to 1"
        )
    return threshold


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
def process_keywords(payload: ProcessInput, request: Request) -> dict[str, object]:
    """Return bilingual subtitles and two ordered, independently served MP3s."""

    public_base_url = _public_base_url(request)

    try:
        keywords = payload.ordered_keywords()
    except RecognitionConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not keywords and payload.results is not None:
        return {"status": "record_again", "record_again": True}

    try:
        sentence_decision = keywords_to_sentence_decision(keywords)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GeminiClientError as exc:
        raise HTTPException(
            status_code=502,
            detail="sentence generation failed",
        ) from exc

    if not sentence_decision.can_complete_sentence:
        return {"status": "record_again", "record_again": True}
    bilingual_text = sentence_decision.bilingual_text
    if bilingual_text is None:
        raise HTTPException(status_code=502, detail="sentence generation failed")

    try:
        speech_segments = text_to_speech_segments(
            bilingual_text.english_sentence,
            bilingual_text.hindi_sentence,
        )
        if len(speech_segments) != 2:
            raise TTSClientError("speech synthesis returned an incomplete result")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TTSClientError as exc:
        raise HTTPException(
            status_code=502,
            detail="speech synthesis failed",
        ) from exc

    file_ids = [uuid.uuid4().hex for _ in speech_segments]
    filenames = [AUDIO_DIR / f"{file_id}.mp3" for file_id in file_ids]
    written_files: list[FilePath] = []
    try:
        for filename, segment in zip(filenames, speech_segments):
            written_files.append(filename)
            filename.write_bytes(segment.audio_content)
    except (OSError, TypeError) as exc:
        for filename in written_files:
            try:
                filename.unlink(missing_ok=True)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail="audio storage failed") from exc

    audio_urls = [
        f"{public_base_url}/audio/{file_id}.mp3" for file_id in file_ids
    ]
    subtitles = [
        {"language": speech_segments[0].language, "text": bilingual_text.english_sentence},
        {"language": speech_segments[1].language, "text": bilingual_text.hindi_sentence},
    ]
    audio_segments = [
        {"language": speech_segments[0].language, "audio_url": audio_urls[0]},
        {"language": speech_segments[1].language, "audio_url": audio_urls[1]},
    ]
    return {
        "sentence": bilingual_text.english_sentence,
        "audio_url": audio_urls[0],
        "hindi_keywords": bilingual_text.hindi_keywords,
        "hindi_sentence": bilingual_text.hindi_sentence,
        "subtitles": subtitles,
        "audio_segments": audio_segments,
        "playback_order": [speech_segments[0].language, speech_segments[1].language],
    }


@router.get("/audio/{file_id}.mp3", response_class=FileResponse)
def audio(file_id: str) -> FileResponse:
    """Serve a previously synthesized local MVP audio file."""

    filename = _audio_file(file_id)
    if not filename.is_file():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(filename, media_type=AUDIO_CONTENT_TYPE)
