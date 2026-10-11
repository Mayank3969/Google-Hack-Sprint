"""Deterministic tests for backend module BE-M7."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

try:
    from backend.services import firestore_client as client
except ModuleNotFoundError:
    from services import firestore_client as client


@pytest.fixture(autouse=True)
def mock_firestore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    client.reset_mock_repository()


def test_mock_room_lifecycle_returns_oldest_first_iso_messages() -> None:
    room_id = client.create_room()

    assert len(room_id) == 32
    assert int(room_id, 16) >= 0

    client.add_message(room_id, "signer", "hello")
    client.add_message(room_id, "speaker", "Hi there")

    messages = client.get_messages(room_id)

    assert [message["sender"] for message in messages] == ["signer", "speaker"]
    assert [message["text"] for message in messages] == ["hello", "Hi there"]
    assert all(message["timestamp"].endswith("Z") for message in messages)
    assert all("T" in message["timestamp"] for message in messages)
    assert messages[0]["timestamp"] <= messages[1]["timestamp"]


def test_mock_repository_reports_missing_rooms() -> None:
    with pytest.raises(client.FirestoreRoomNotFoundError):
        client.add_message("missing-room", "signer", "hello")

    with pytest.raises(client.FirestoreRoomNotFoundError):
        client.get_messages("missing-room")


@pytest.mark.parametrize(
    "room_id, sender, text",
    [
        ("", "signer", "hello"),
        ("room", "", "hello"),
        ("room", "signer", ""),
        ("room", 1, "hello"),
    ],
)
def test_message_storage_rejects_empty_or_non_string_values(
    room_id: object,
    sender: object,
    text: object,
) -> None:
    with pytest.raises(client.FirestoreClientError):
        client.add_message(room_id, sender, text)  # type: ignore[arg-type]


def test_mock_reset_removes_rooms() -> None:
    room_id = client.create_room()
    client.reset_mock_repository()

    with pytest.raises(client.FirestoreRoomNotFoundError):
        client.get_messages(room_id)


def test_mock_mode_does_not_load_firestore_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client,
        "_load_firestore",
        lambda: (_ for _ in ()).throw(AssertionError("Firestore must stay lazy")),
    )

    room_id = client.create_room()
    client.add_message(room_id, "signer", "mock only")

    assert client.get_messages(room_id)[0]["text"] == "mock only"


def test_serialize_message_normalizes_timestamp_to_utc_z() -> None:
    timestamp = datetime(2026, 1, 1, 5, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    result = client.serialize_message(
        {"sender": "speaker", "text": "hello", "timestamp": timestamp}
    )

    assert result == {
        "sender": "speaker",
        "text": "hello",
        "timestamp": "2026-01-01T00:00:00Z",
    }


def test_serialize_message_accepts_provider_timestamp_method() -> None:
    timestamp = SimpleNamespace(
        ToDatetime=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    result = client.serialize_message(
        {"sender": "signer", "text": "hello", "timestamp": timestamp}
    )

    assert result["timestamp"] == "2026-01-01T00:00:00Z"


class FakeDocument:
    def __init__(self, room_id: str, rooms: dict[str, bool], messages: dict[str, list[dict[str, object]]]) -> None:
        self.room_id = room_id
        self.rooms = rooms
        self.messages = messages
        self.created: dict[str, object] | None = None

    def set(self, values: dict[str, object]) -> None:
        self.rooms[self.room_id] = True
        self.created = values
        self.messages.setdefault(self.room_id, [])

    def get(self) -> SimpleNamespace:
        return SimpleNamespace(exists=self.rooms.get(self.room_id, False))

    def collection(self, name: str) -> "FakeMessageCollection":
        assert name == "messages"
        return FakeMessageCollection(self.messages.setdefault(self.room_id, []))


class FakeMessageCollection:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages = messages

    def add(self, values: dict[str, object]) -> None:
        self.messages.append(values)

    def order_by(self, field: str) -> "FakeMessageCollection":
        assert field == "timestamp"
        return self

    def stream(self) -> list[SimpleNamespace]:
        ordered = sorted(self.messages, key=lambda message: message["timestamp"])
        return [SimpleNamespace(to_dict=lambda message=message: message) for message in ordered]


class FakeFirestoreDatabase:
    def __init__(self) -> None:
        self.rooms: dict[str, bool] = {}
        self.messages: dict[str, list[dict[str, object]]] = {}

    def collection(self, name: str) -> "FakeRoomCollection":
        assert name == "rooms"
        return FakeRoomCollection(self.rooms, self.messages)


class FakeRoomCollection:
    def __init__(self, rooms: dict[str, bool], messages: dict[str, list[dict[str, object]]]) -> None:
        self.rooms = rooms
        self.messages = messages

    def document(self, room_id: str) -> FakeDocument:
        return FakeDocument(room_id, self.rooms, self.messages)


def test_real_firestore_shape_is_lazy_and_serializes_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = FakeFirestoreDatabase()
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(client, "_db", None)
    monkeypatch.setattr(client, "_get_database", lambda: database)

    room_id = client.create_room()
    client.add_message(room_id, "signer", "hello")
    messages = client.get_messages(room_id)

    assert len(room_id) == 32
    assert messages[0]["sender"] == "signer"
    assert messages[0]["text"] == "hello"
    assert messages[0]["timestamp"].endswith("Z")


def test_real_firestore_missing_room_is_controlled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(client, "_get_database", lambda: FakeFirestoreDatabase())

    with pytest.raises(client.FirestoreRoomNotFoundError):
        client.get_messages("missing-room")


def test_real_firestore_provider_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenDatabase:
        def collection(self, name: str) -> None:
            raise RuntimeError("provider details should not escape")

    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(client, "_get_database", lambda: BrokenDatabase())

    with pytest.raises(client.FirestoreProviderError) as error:
        client.create_room()

    assert str(error.value) == "Firestore provider operation failed"
