"""HTTP-level tests for backend module BE-M8."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

try:
    from backend.main import app
    from backend.routes import session as session_route
    from backend.services import firestore_client
except ModuleNotFoundError:
    from main import app
    from routes import session as session_route
    from services import firestore_client


client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_firestore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")
    firestore_client.reset_mock_repository()


def test_session_lifecycle_returns_both_messages_oldest_first() -> None:
    create_response = client.post("/session/create")

    assert create_response.status_code == 200
    room_id = create_response.json()["room_id"]
    assert len(room_id) == 32

    signer_response = client.post(
        f"/session/{room_id}/message",
        json={"sender": "signer", "text": "  Can I pay?  "},
    )
    speaker_response = client.post(
        f"/session/{room_id}/message",
        json={"sender": "speaker", "text": "Yes, that is fine."},
    )

    assert signer_response.status_code == 200
    assert signer_response.json() == {"status": "ok"}
    assert speaker_response.status_code == 200
    assert speaker_response.json() == {"status": "ok"}

    messages_response = client.get(f"/session/{room_id}/messages")

    assert messages_response.status_code == 200
    messages = messages_response.json()["messages"]
    assert [message["sender"] for message in messages] == ["signer", "speaker"]
    assert [message["text"] for message in messages] == [
        "Can I pay?",
        "Yes, that is fine.",
    ]
    assert all(message["timestamp"].endswith("Z") for message in messages)


@pytest.mark.parametrize(
    "payload",
    [
        {"sender": "unknown", "text": "hello"},
        {"sender": 1, "text": "hello"},
        {"sender": "signer", "text": "   "},
        {"sender": "signer", "text": "x" * 2001},
        {"sender": "signer"},
        {},
    ],
)
def test_session_message_rejects_invalid_payload(payload: dict[str, object]) -> None:
    room_id = client.post("/session/create").json()["room_id"]

    response = client.post(f"/session/{room_id}/message", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_session_message_rejects_unknown_room() -> None:
    response = client.post(
        "/session/missing-room/message",
        json={"sender": "signer", "text": "hello"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation room not found"}


def test_session_messages_rejects_unknown_room() -> None:
    response = client.get("/session/missing-room/messages")

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation room not found"}


def test_session_routes_reject_oversized_room_id() -> None:
    room_id = "r" * 129

    post_response = client.post(
        f"/session/{room_id}/message",
        json={"sender": "signer", "text": "hello"},
    )
    get_response = client.get(f"/session/{room_id}/messages")

    assert post_response.status_code == 422
    assert get_response.status_code == 422


def test_session_create_maps_storage_failure_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_route.firestore_client,
        "create_room",
        lambda: (_ for _ in ()).throw(
            firestore_client.FirestoreProviderError("provider details should not escape")
        ),
    )

    response = client.post("/session/create")

    assert response.status_code == 502
    assert response.json() == {"detail": "session storage failed"}


def test_session_message_maps_storage_failure_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room_id = client.post("/session/create").json()["room_id"]
    monkeypatch.setattr(
        session_route.firestore_client,
        "add_message",
        lambda room, sender, text: (_ for _ in ()).throw(
            firestore_client.FirestoreProviderError("provider details should not escape")
        ),
    )

    response = client.post(
        f"/session/{room_id}/message",
        json={"sender": "signer", "text": "hello"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "session storage failed"}


def test_session_messages_maps_storage_failure_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room_id = client.post("/session/create").json()["room_id"]
    monkeypatch.setattr(
        session_route.firestore_client,
        "get_messages",
        lambda room: (_ for _ in ()).throw(
            firestore_client.FirestoreProviderError("provider details should not escape")
        ),
    )

    response = client.get(f"/session/{room_id}/messages")

    assert response.status_code == 502
    assert response.json() == {"detail": "session storage failed"}


def test_session_messages_reject_malformed_storage_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room_id = client.post("/session/create").json()["room_id"]
    monkeypatch.setattr(
        session_route.firestore_client,
        "get_messages",
        lambda room: [{"sender": "signer", "text": "missing timestamp"}],
    )

    response = client.get(f"/session/{room_id}/messages")

    assert response.status_code == 502
    assert response.json() == {"detail": "session storage failed"}
