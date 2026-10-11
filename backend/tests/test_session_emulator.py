"""Opt-in HTTP lifecycle check against a loopback Firestore emulator only."""

import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import firestore_client


def test_emulator_session_lifecycle(monkeypatch):
    if os.getenv("RUN_FIRESTORE_EMULATOR_TESTS") != "1":
        pytest.skip("Run with Firebase emulators:exec and explicit opt-in")
    host = os.getenv("FIRESTORE_EMULATOR_HOST", "")
    assert host == "127.0.0.1:8082", "Refusing non-local emulator target"
    assert os.getenv("GOOGLE_CLOUD_PROJECT") == "demo-signbridge"
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "0")
    monkeypatch.setattr(firestore_client, "_db", None)
    with TestClient(app) as client:
        created = client.post("/session/create")
        assert created.status_code == 200
        room = created.json()["room_id"]
        assert client.get(f"/session/{room}/messages").json() == {"messages": []}
        for sender in ("signer", "speaker"):
            response = client.post(f"/session/{room}/message", json={
                "sender": sender, "text": f"Synthetic {sender} test"
            })
            assert response.status_code == 200
        result = client.get(f"/session/{room}/messages")
        assert result.status_code == 200
        messages = result.json()["messages"]
        assert [m["sender"] for m in messages] == ["signer", "speaker"]
        assert all(m["timestamp"].endswith("Z") for m in messages)
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in messages]
        assert timestamps == sorted(timestamps)
        assert client.get("/session/nonexistent-test-room/messages").status_code == 404
        assert client.post(f"/session/{room}/message", json={
            "sender": "invalid", "text": "test"
        }).status_code == 422
