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
    monkeypatch.delenv("FIRESTORE_TIMEOUT_SECONDS", raising=False)
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
    def __init__(self, room_id: str, rooms: dict[str, bool], messages: dict[str, list[dict[str, object]]], calls: list) -> None:
        self.room_id = room_id
        self.rooms = rooms
        self.messages = messages
        self.created: dict[str, object] | None = None
        self.calls = calls

    def set(self, values: dict[str, object], **kwargs) -> None:
        self.calls.append(("set", kwargs))
        self.rooms[self.room_id] = True
        self.created = values
        self.messages.setdefault(self.room_id, [])

    def get(self, **kwargs) -> SimpleNamespace:
        self.calls.append(("get", kwargs))
        return SimpleNamespace(exists=self.rooms.get(self.room_id, False))

    def collection(self, name: str) -> "FakeMessageCollection":
        assert name == "messages"
        return FakeMessageCollection(self.messages.setdefault(self.room_id, []), self.calls)


class FakeMessageCollection:
    def __init__(self, messages: list[dict[str, object]], calls: list) -> None:
        self.messages = messages
        self.calls = calls

    def add(self, values: dict[str, object], **kwargs) -> None:
        self.calls.append(("add", kwargs))
        self.messages.append(values)

    def order_by(self, field: str) -> "FakeMessageCollection":
        assert field == "timestamp"
        return self

    def stream(self, **kwargs) -> list[SimpleNamespace]:
        self.calls.append(("stream", kwargs))
        ordered = sorted(self.messages, key=lambda message: message["timestamp"])
        return [SimpleNamespace(to_dict=lambda message=message: message) for message in ordered]


class FakeFirestoreDatabase:
    def __init__(self) -> None:
        self.rooms: dict[str, bool] = {}
        self.messages: dict[str, list[dict[str, object]]] = {}
        self.calls: list = []

    def collection(self, name: str) -> "FakeRoomCollection":
        assert name == "rooms"
        return FakeRoomCollection(self.rooms, self.messages, self.calls)


class FakeRoomCollection:
    def __init__(self, rooms: dict[str, bool], messages: dict[str, list[dict[str, object]]], calls: list) -> None:
        self.rooms = rooms
        self.messages = messages
        self.calls = calls

    def document(self, room_id: str) -> FakeDocument:
        return FakeDocument(room_id, self.rooms, self.messages, self.calls)


def test_real_firestore_shape_is_lazy_and_serializes_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = FakeFirestoreDatabase()
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("FIRESTORE_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setattr(client, "_db", None)
    monkeypatch.setattr(client, "_get_database", lambda: database)

    room_id = client.create_room()
    client.add_message(room_id, "signer", "hello")
    messages = client.get_messages(room_id)

    assert len(room_id) == 32
    assert messages[0]["sender"] == "signer"
    assert messages[0]["text"] == "hello"
    assert messages[0]["timestamp"].endswith("Z")
    assert database.calls == [
        (method, {"timeout": 12.5, "retry": None})
        for method in ("set", "get", "add", "get", "stream")
    ]


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


@pytest.mark.parametrize("value", ["", " ", "bad", "nan", "inf", "-inf", "0", "-1", "0.5", "60.1"])
@pytest.mark.parametrize("operation", ["create", "add", "get"])
def test_invalid_timeout_fails_before_provider_setup(monkeypatch, value, operation) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setenv("FIRESTORE_TIMEOUT_SECONDS", value)
    monkeypatch.setattr(
        client, "_get_database",
        lambda: pytest.fail("invalid settings must not initialize the provider"),
    )
    with pytest.raises(client.FirestoreConfigurationError, match="FIRESTORE_TIMEOUT_SECONDS"):
        if operation == "create":
            client.create_room()
        elif operation == "add":
            client.add_message("room", "signer", "hello")
        else:
            client.get_messages("room")


@pytest.mark.parametrize("value, expected", [(None, 15.0), ("1", 1.0), ("60", 60.0)])
def test_timeout_defaults_and_inclusive_bounds(monkeypatch, value, expected) -> None:
    if value is not None:
        monkeypatch.setenv("FIRESTORE_TIMEOUT_SECONDS", value)
    assert client._timeout_seconds_from_environment() == expected


@pytest.mark.parametrize("method", ["set", "get", "add", "stream"])
def test_rpc_timeout_is_wrapped_without_retry(monkeypatch, method) -> None:
    database = FakeFirestoreDatabase()
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(client, "_get_database", lambda: database)
    room = client.create_room()
    calls = []

    def timed_out(self, *args, **kwargs):
        calls.append(kwargs)
        raise TimeoutError("private provider details")

    owner = FakeDocument if method in {"set", "get"} else FakeMessageCollection
    monkeypatch.setattr(owner, method, timed_out)
    with pytest.raises(client.FirestoreProviderError) as error:
        if method == "set":
            client.create_room()
        elif method == "add":
            client.add_message(room, "signer", "hello")
        else:
            client.get_messages(room)
    assert str(error.value) == "Firestore provider operation failed"
    assert calls == [{"timeout": 15.0, "retry": None}]
