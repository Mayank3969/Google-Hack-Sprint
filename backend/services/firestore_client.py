"""Small Firestore-backed conversation store used by the backend routes.

The provider is deliberately loaded lazily.  This keeps health checks and mock
mode usable on machines that do not have the Google Cloud client installed or
configured, while still making real-provider failures explicit.
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping


class FirestoreClientError(RuntimeError):
    """Base class for safe, public Firestore client failures."""


class FirestoreConfigurationError(FirestoreClientError):
    """The Firestore provider or its local credentials are not configured."""


class FirestoreProviderError(FirestoreClientError):
    """A Firestore provider operation failed."""


class FirestoreRoomNotFoundError(FirestoreClientError):
    """The requested conversation room does not exist."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FirestoreClientError(f"{name} must be a non-empty string")
    return value


class MockFirestoreRepository:
    """In-memory repository used only when mock mode is explicitly enabled."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        """Remove all mock rooms and messages.

        This is intentionally public so tests and local demos can start from a
        clean repository without reaching into implementation details.
        """

        with getattr(self, "_lock", threading.RLock()):
            self.rooms: dict[str, datetime] = {}
            self.messages: dict[str, list[tuple[datetime, int, dict[str, Any]]]] = {}
            self._sequence = 0

    def create_room(self) -> str:
        room_id = uuid.uuid4().hex
        created_at = _utc_now()
        with self._lock:
            self.rooms[room_id] = created_at
            self.messages[room_id] = []
        return room_id

    def add_message(self, room_id: str, sender: str, text: str) -> None:
        with self._lock:
            if room_id not in self.rooms:
                raise FirestoreRoomNotFoundError("conversation room was not found")
            timestamp = _utc_now()
            sequence = self._sequence
            self._sequence += 1
            self.messages[room_id].append(
                (
                    timestamp,
                    sequence,
                    {"sender": sender, "text": text, "timestamp": timestamp},
                )
            )

    def get_messages(self, room_id: str) -> list[dict[str, Any]]:
        with self._lock:
            if room_id not in self.rooms:
                raise FirestoreRoomNotFoundError("conversation room was not found")
            entries = sorted(self.messages[room_id], key=lambda item: (item[0], item[1]))
            return [dict(entry[2]) for entry in entries]


_mock_repository = MockFirestoreRepository()


def reset_mock_repository() -> None:
    """Reset the process-local mock repository for tests or local demos."""

    _mock_repository.reset()


def _mock_mode_enabled() -> bool:
    value = os.getenv("MOCK_EXTERNAL_SERVICES")
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


# The real provider is not imported or constructed until this getter is used.
_db: Any | None = None
_db_lock = threading.RLock()


def _load_firestore() -> Any:
    """Load the Firestore module only when real mode is selected."""

    try:
        from google.cloud import firestore
    except Exception as exc:  # ImportError and broken provider installations.
        raise FirestoreConfigurationError(
            "Firestore provider is not available"
        ) from exc
    return firestore


def _create_real_client(firestore_module: Any) -> Any:
    """Construct a Firestore client using Application Default Credentials."""

    return firestore_module.Client()


def _get_database() -> Any:
    """Return the lazily-created real Firestore client."""

    global _db
    if _db is not None:
        return _db
    with _db_lock:
        if _db is not None:
            return _db
        try:
            firestore_module = _load_firestore()
            _db = _create_real_client(firestore_module)
        except FirestoreClientError:
            raise
        except Exception as exc:
            raise FirestoreConfigurationError(
                "Firestore client could not be configured"
            ) from exc
        return _db


def _provider_failure(exc: Exception) -> FirestoreProviderError:
    return FirestoreProviderError("Firestore provider operation failed")


def _room_exists(snapshot: Any) -> bool:
    exists = getattr(snapshot, "exists", False)
    return bool(exists() if callable(exists) else exists)


def _get_real_room(db: Any, room_id: str) -> Any:
    try:
        room = db.collection("rooms").document(room_id)
        snapshot = room.get()
    except FirestoreClientError:
        raise
    except Exception as exc:
        raise _provider_failure(exc) from exc
    try:
        exists = _room_exists(snapshot)
    except Exception as exc:
        raise _provider_failure(exc) from exc
    if not exists:
        raise FirestoreRoomNotFoundError("conversation room was not found")
    return room


def _timestamp_as_datetime(value: Any) -> datetime:
    """Convert supported provider timestamp values to aware UTC datetime."""

    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise FirestoreProviderError("message timestamp is invalid")
        try:
            result = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise FirestoreProviderError("message timestamp is invalid") from exc
    else:
        result = None
        for method_name in ("ToDatetime", "to_datetime"):
            method = getattr(value, method_name, None)
            if not callable(method):
                continue
            try:
                result = method()
            except Exception as exc:
                raise FirestoreProviderError("message timestamp is invalid") from exc
            break
        if result is None:
            raise FirestoreProviderError("message timestamp is invalid")

    if not isinstance(result, datetime):
        raise FirestoreProviderError("message timestamp is invalid")
    try:
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        else:
            result = result.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FirestoreProviderError("message timestamp is invalid") from exc
    return result


def _timestamp_as_iso(value: Any) -> str:
    timestamp = _timestamp_as_datetime(value)
    try:
        return timestamp.isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError) as exc:
        raise FirestoreProviderError("message timestamp is invalid") from exc


def serialize_message(message: Mapping[str, Any]) -> dict[str, str]:
    """Return a provider-neutral message suitable for the HTTP API boundary."""

    if not isinstance(message, Mapping):
        raise FirestoreProviderError("message data is invalid")
    try:
        sender = message["sender"]
        text = message["text"]
        timestamp = message["timestamp"]
    except (KeyError, TypeError) as exc:
        raise FirestoreProviderError("message data is invalid") from exc
    if not isinstance(sender, str) or not sender:
        raise FirestoreProviderError("message sender is invalid")
    if not isinstance(text, str) or not text:
        raise FirestoreProviderError("message text is invalid")
    return {
        "sender": sender,
        "text": text,
        "timestamp": _timestamp_as_iso(timestamp),
    }


def create_room() -> str:
    """Create and return a cryptographically non-trivial room identifier."""

    if _mock_mode_enabled():
        return _mock_repository.create_room()

    db = _get_database()
    room_id = uuid.uuid4().hex
    try:
        db.collection("rooms").document(room_id).set({"created_at": _utc_now()})
    except FirestoreClientError:
        raise
    except Exception as exc:
        raise _provider_failure(exc) from exc
    return room_id


def add_message(room_id: str, sender: str, text: str) -> None:
    """Store one message, rejecting only values invalid for storage itself."""

    room_id = _require_nonempty_string(room_id, "room_id")
    sender = _require_nonempty_string(sender, "sender")
    text = _require_nonempty_string(text, "text")
    if _mock_mode_enabled():
        _mock_repository.add_message(room_id, sender, text)
        return

    db = _get_database()
    room = _get_real_room(db, room_id)
    try:
        room.collection("messages").add(
            {"sender": sender, "text": text, "timestamp": _utc_now()}
        )
    except FirestoreClientError:
        raise
    except Exception as exc:
        raise _provider_failure(exc) from exc


def get_messages(room_id: str) -> list[dict[str, str]]:
    """Return room messages oldest first with UTC ISO-8601 timestamps."""

    room_id = _require_nonempty_string(room_id, "room_id")
    if _mock_mode_enabled():
        return [serialize_message(message) for message in _mock_repository.get_messages(room_id)]

    db = _get_database()
    room = _get_real_room(db, room_id)
    try:
        documents = room.collection("messages").order_by("timestamp").stream()
        messages = []
        for document in documents:
            raw_message = document.to_dict()
            messages.append(serialize_message(raw_message))
        return messages
    except FirestoreClientError:
        raise
    except Exception as exc:
        raise _provider_failure(exc) from exc


# A short private alias is useful to test suites that prefer a fixture-style
# reset without making the singleton itself part of the public API.
_reset_mock_repository = reset_mock_repository
