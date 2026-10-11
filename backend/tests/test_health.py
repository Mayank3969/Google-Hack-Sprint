"""HTTP-level checks for backend module BE-M1."""

from fastapi.testclient import TestClient

try:
    from backend.main import app
except ModuleNotFoundError:
    # Allows pytest to run from inside ``backend/`` as documented.
    from main import app


client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
