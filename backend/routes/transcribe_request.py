"""Bound and sanitize request bodies for the transcribe route only."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from json import JSONDecodeError

from fastapi import HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.requests import ClientDisconnect
from starlette.types import Message

if __package__ and __package__.startswith("backend."):
    from ..services.audio_validation import MAX_AUDIO_BYTES
else:  # Supports imports when Uvicorn is started from the backend directory.
    from services.audio_validation import MAX_AUDIO_BYTES


MAX_BASE64_AUDIO_CHARS = 4 * ((MAX_AUDIO_BYTES + 2) // 3)
MAX_TRANSCRIBE_BODY_BYTES = MAX_BASE64_AUDIO_CHARS + 4096

_JSON_MEDIA_TYPE = re.compile(
    r"application/(?:json|[!#$%&'*+.^_`|~0-9a-z-]+\+json)"
)
_BODY_TOO_LARGE = "request body exceeds the maximum size"
_INVALID_REQUEST = "request must contain an audio string in valid JSON"


class TranscribeRequestRoute(APIRoute):
    """Apply a raw-body limit before FastAPI performs request validation."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        original_handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            content_types = request.headers.getlist("content-type")
            if len(content_types) != 1 or not _JSON_MEDIA_TYPE.fullmatch(
                content_types[0].split(";", 1)[0].strip().lower()
            ):
                raise HTTPException(
                    status_code=415, detail="request Content-Type must be JSON"
                )
            if any(
                encoding.strip().lower() != "identity"
                for encoding in request.headers.getlist("content-encoding")
            ):
                raise HTTPException(
                    status_code=415, detail="request Content-Encoding must be identity"
                )

            content_lengths = request.headers.getlist("content-length")
            if content_lengths:
                if len(content_lengths) != 1 or not re.fullmatch(
                    r"[0-9]+", content_lengths[0]
                ):
                    raise HTTPException(status_code=400, detail="invalid Content-Length")
                # Compare decimal strings to avoid integer conversion limits.
                length = content_lengths[0].lstrip("0") or "0"
                maximum = str(MAX_TRANSCRIBE_BODY_BYTES)
                if len(length) > len(maximum) or (
                    len(length) == len(maximum) and length > maximum
                ):
                    raise HTTPException(status_code=413, detail=_BODY_TOO_LARGE)

            body = bytearray()
            try:
                async for chunk in request.stream():
                    if len(chunk) > MAX_TRANSCRIBE_BODY_BYTES - len(body):
                        raise HTTPException(status_code=413, detail=_BODY_TOO_LARGE)
                    body.extend(chunk)
            except ClientDisconnect:
                raise HTTPException(
                    status_code=400, detail="request body was not received completely"
                ) from None

            replayed = False

            async def replay_receive() -> Message:
                nonlocal replayed
                if replayed:
                    return {"type": "http.disconnect"}
                replayed = True
                payload = bytes(body)
                body.clear()
                return {"type": "http.request", "body": payload, "more_body": False}

            bounded_request = Request(request.scope, receive=replay_receive)
            # Cache JSON through the public API and isolate parser failures from
            # errors raised later by dependencies or the endpoint itself.
            try:
                await bounded_request.json()
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400, detail="invalid JSON request body"
                ) from None
            except JSONDecodeError:
                raise HTTPException(status_code=422, detail=_INVALID_REQUEST) from None
            except RecursionError:
                raise HTTPException(
                    status_code=400, detail="invalid JSON request body"
                ) from None

            try:
                return await original_handler(bounded_request)
            except RequestValidationError:
                raise HTTPException(status_code=422, detail=_INVALID_REQUEST) from None

        return route_handler
