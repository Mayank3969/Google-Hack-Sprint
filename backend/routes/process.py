"""Keyword-to-bilingual-text-and-audio routes for backend module BE-M11."""

from __future__ import annotations

import math
import os
import re
import tempfile
import uuid
from ipaddress import IPv6Address
from pathlib import Path as FilePath
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from typing import Any, Literal

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
MAX_TOP_K_RESULTS = 10
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
    confidence: StrictFloat = Field(ge=0.0, le=1.0)


class RecognitionResult(BaseModel):
    """One segmented ML recognition result."""

    segment_index: StrictInt = Field(ge=0, lt=MAX_CAPTURE_RESULTS)
    status: Literal[
        "recognized",
        "uncertain",
        "insufficient_data",
        "tracking_lost",
        "invalid",
    ]
    keyword: StrictStr | None = Field(default=None, max_length=MAX_KEYWORD_LENGTH)
    confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    # The adapter used by the frontend sends ``coverage`` directly.  The ML v2
    # service puts the same value in ``quality.coordinate_coverage`` instead,
    # so both shapes are accepted at this boundary.
    coverage: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    top_k: list[TopKResult] | None = Field(default=None, max_length=MAX_TOP_K_RESULTS)
    quality: dict[str, Any] | None = None


class ProcessInput(BaseModel):
    """Validate legacy keywords or the segmented ML capture-results payload."""

    keywords: list[StrictStr] | None = Field(default=None, max_length=MAX_KEYWORDS)
    capture_id: StrictStr | None = Field(default=None, min_length=1, max_length=128)
    results: list[RecognitionResult] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_CAPTURE_RESULTS,
        description="Ordered or indexed segmented ML recognition results.",
    )
    # These fields are emitted by /v2/predict/batch.  The frontend adapter may
    # omit them and rely on the per-slot statuses; omitted values therefore do
    # not weaken the all-slots-recognized check below.
    complete: StrictBool | None = None
    integration_ready: StrictBool | None = None
    locale: Literal["en-IN", "hi-IN"] | None = None

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
        if not has_legacy and not has_capture:
            raise ValueError("provide keywords or capture_id/results")
        # ``keywords`` is part of the full ML v2 response envelope, but a
        # legacy request must not smuggle a second input shape alongside a
        # capture.  The envelope flags distinguish those two cases.
        if (
            has_legacy
            and has_capture
            and self.complete is None
            and self.integration_ready is None
        ):
            raise ValueError("provide either keywords or capture_id/results")
        if not has_capture:
            if not self.keywords:
                raise ValueError("keywords must not be empty")
            return self
        if self.capture_id is None or self.results is None:
            raise ValueError("capture_id and results are required together")
        if len({result.segment_index for result in self.results}) != len(self.results):
            raise ValueError("results must contain unique segment_index values")
        return self

    @staticmethod
    def _quality_coverage(result: RecognitionResult) -> float | None:
        """Read ML v2 coverage without accepting arbitrary quality metadata."""

        if result.coverage is not None:
            return result.coverage
        if result.quality is None:
            return None
        value = result.quality.get("coordinate_coverage")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            return None
        return float(value)

    def batch_is_complete(self) -> bool:
        """Return whether the complete capture is safe to send to Gemini.

        A batch is all-or-nothing.  In particular, a recognized prefix must not
        be sent when another slot is uncertain/invalid: the slots represent one
        phrase and repeated labels are meaningful signs, not values to dedupe.
        """

        if self.results is None:
            return True
        if self.complete is False or self.integration_ready is False:
            return False

        threshold = _recognition_threshold()
        for result in self.results:
            if result.status != "recognized":
                return False
            if not result.keyword or not result.keyword.strip():
                return False
            if result.confidence is None or result.confidence < threshold:
                return False
            coverage = self._quality_coverage(result)
            if coverage is None or coverage < threshold:
                return False

        if self.complete is True and self.keywords is not None:
            expected = [
                result.keyword.strip()
                for result in sorted(self.results, key=lambda item: item.segment_index)
                if result.keyword
            ]
            if self.keywords != expected:
                return False
        return bool(self.results)

    def ordered_keywords(self) -> list[str]:
        """Return labels in segment order, retaining legitimate duplicates."""

        if self.results is None:
            assert self.keywords is not None
            return self.keywords
        if not self.batch_is_complete():
            return []
        selected = [
            result.keyword.strip()
            for result in sorted(self.results, key=lambda result: result.segment_index)
            if result.keyword is not None
        ]
        if len(selected) > MAX_KEYWORDS:
            raise ValueError(
                f"recognized keywords must contain at most {MAX_KEYWORDS} items"
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
        if payload.results is not None and not payload.batch_is_complete():
            return {"status": "record_again", "record_again": True}
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
    selected_index = 1 if payload.locale == "hi-IN" else 0
    playback_order = [
        speech_segments[selected_index].language,
        speech_segments[1 - selected_index].language,
    ]
    selected_sentence = (
        bilingual_text.hindi_sentence
        if selected_index == 1
        else bilingual_text.english_sentence
    )
    selected_audio_url = audio_urls[selected_index]
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
        "playback_order": playback_order,
        "selected_language": payload.locale or "en-IN",
        "selected_sentence": selected_sentence,
        "selected_audio_url": selected_audio_url,
    }


@router.get("/audio/{file_id}.mp3", response_class=FileResponse)
def audio(file_id: str) -> FileResponse:
    """Serve a previously synthesized local MVP audio file."""

    filename = _audio_file(file_id)
    if not filename.is_file():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(filename, media_type=AUDIO_CONTENT_TYPE)
