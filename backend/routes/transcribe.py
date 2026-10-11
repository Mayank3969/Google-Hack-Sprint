"""Bounded speech-to-text route for backend module BE-M6."""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, StrictStr

from .transcribe_request import TranscribeRequestRoute, MAX_BASE64_AUDIO_CHARS

if __package__ and __package__.startswith("backend."):
    from ..services import stt_client
    from ..services.audio_validation import MAX_AUDIO_BYTES, AudioValidationError
else:  # Supports imports when Uvicorn is started from the backend directory.
    from services import stt_client
    from services.audio_validation import MAX_AUDIO_BYTES, AudioValidationError


router = APIRouter(tags=["transcribe"], route_class=TranscribeRequestRoute)


class TranscribeInput(BaseModel):
    """A strictly base64-encoded WAV recording."""

    audio: StrictStr


class TranscribeOutput(BaseModel):
    """Recognized text, including an empty string when no speech is recognized."""

    text: str


@router.post(
    "/transcribe",
    response_model=TranscribeOutput,
    summary="Transcribe a bounded WAV recording",
    description=(
        "Accepts strictly base64-encoded mono 16000 Hz 16-bit PCM WAV audio "
        f"of at most 30 seconds, {MAX_AUDIO_BYTES} decoded bytes, and "
        f"{MAX_BASE64_AUDIO_CHARS} base64 characters. Silence or no recognized "
        "speech returns an empty text string. No automatic session writes."
    ),
)
def transcribe(payload: TranscribeInput) -> TranscribeOutput:
    """Transcribe bounded audio; silence returns empty text without session writes."""

    if len(payload.audio) > MAX_BASE64_AUDIO_CHARS:
        raise HTTPException(status_code=413, detail="audio exceeds the maximum size")

    try:
        raw = base64.b64decode(payload.audio, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="audio must be valid base64") from None

    if not raw:
        raise HTTPException(status_code=400, detail="audio must not be empty")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio exceeds the maximum size")

    try:
        text = stt_client.speech_to_text(raw)
    except AudioValidationError:
        raise HTTPException(
            status_code=400,
            detail=(
                "audio must be a valid mono 16000 Hz 16-bit PCM WAV "
                "of at most 30 seconds"
            ),
        ) from None
    except stt_client.STTClientError:
        raise HTTPException(
            status_code=502, detail="speech transcription failed"
        ) from None

    if not isinstance(text, str):
        raise HTTPException(
            status_code=502, detail="speech transcription failed"
        ) from None

    return TranscribeOutput(text=text)
