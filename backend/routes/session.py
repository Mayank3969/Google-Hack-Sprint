"""Conversation session routes for backend module BE-M8."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field, StrictStr, ValidationError, field_validator

if __package__ and __package__.startswith("backend."):
    from ..services import firestore_client
else:  # Supports imports when Uvicorn is started from the backend directory.
    from services import firestore_client


router = APIRouter(tags=["session"])

_MAX_ROOM_ID_LENGTH = 128


class CreateSessionResponse(BaseModel):
    room_id: StrictStr


class MessageInput(BaseModel):
    """Validated message input for the conversation timeline."""

    sender: Literal["signer", "speaker"]
    text: StrictStr = Field(
        min_length=1,
        max_length=2000,
        description="Message text; surrounding whitespace is removed. Maximum 2000 characters.",
    )

    @field_validator("sender", mode="before")
    @classmethod
    def require_string_sender(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("sender must be a string")
        return value

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("text must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be blank")
        return cleaned


class SessionMessage(BaseModel):
    sender: Literal["signer", "speaker"]
    text: StrictStr = Field(min_length=1, max_length=2000)
    timestamp: StrictStr


class MessagesResponse(BaseModel):
    messages: list[SessionMessage]


def _validate_room_id(room_id: str) -> str:
    """Reject blank or oversized room path values with a controlled client error."""

    if not room_id.strip():
        raise HTTPException(status_code=422, detail="room_id must not be blank")
    if len(room_id) > _MAX_ROOM_ID_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"room_id must be at most {_MAX_ROOM_ID_LENGTH} characters",
        )
    return room_id


def _storage_failure() -> HTTPException:
    return HTTPException(status_code=502, detail="session storage failed")


@router.post("/session/create", response_model=CreateSessionResponse)
def session_create() -> dict[str, str]:
    """Create a conversation room."""

    try:
        room_id = firestore_client.create_room()
    except firestore_client.FirestoreClientError:
        raise _storage_failure() from None
    return {"room_id": room_id}


@router.post("/session/{room_id}/message")
def session_message(
    room_id: str = Path(
        min_length=1,
        max_length=_MAX_ROOM_ID_LENGTH,
        description="Conversation room identifier (maximum 128 characters).",
    ),
    payload: MessageInput = ...,
) -> dict[str, str]:
    """Append a message; text is a strict string with a maximum length of 2000."""

    room_id = _validate_room_id(room_id)
    try:
        firestore_client.add_message(room_id, payload.sender, payload.text)
    except firestore_client.FirestoreRoomNotFoundError:
        raise HTTPException(
            status_code=404, detail="conversation room not found"
        ) from None
    except firestore_client.FirestoreClientError:
        raise _storage_failure() from None
    return {"status": "ok"}


@router.get("/session/{room_id}/messages", response_model=MessagesResponse)
def session_messages(
    room_id: str = Path(
        min_length=1,
        max_length=_MAX_ROOM_ID_LENGTH,
        description="Conversation room identifier (maximum 128 characters).",
    ),
) -> dict[str, list[dict[str, str]]]:
    """Return messages in the Firestore client's oldest-first order."""

    room_id = _validate_room_id(room_id)
    try:
        messages = firestore_client.get_messages(room_id)
        if not isinstance(messages, list) or any(
            not isinstance(message, dict) for message in messages
        ):
            raise _storage_failure()
        MessagesResponse.model_validate({"messages": messages})
    except firestore_client.FirestoreRoomNotFoundError:
        raise HTTPException(
            status_code=404, detail="conversation room not found"
        ) from None
    except firestore_client.FirestoreClientError:
        raise _storage_failure() from None
    except ValidationError:
        raise _storage_failure() from None
    return {"messages": messages}
