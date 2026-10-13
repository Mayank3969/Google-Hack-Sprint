"""Browser-facing CORS policy and startup validation checks."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

try:
    from backend.main import create_app
except ModuleNotFoundError:
    from main import create_app


def make_client(monkeypatch, origins=None):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    if origins is not None:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", origins)
    return TestClient(create_app())


def preflight(client, origin, method="POST", headers="Content-Type"):
    return client.options("/health", headers={
        "Origin": origin,
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": headers,
    })


def test_allowed_preflight_and_actual_request(monkeypatch):
    client = make_client(monkeypatch, " http://localhost:5173, https://app.example.com ")
    for origin in ("http://localhost:5173", "https://app.example.com"):
        response = preflight(client, origin)
        assert response.status_code == 200
        assert set(response.headers["access-control-allow-methods"].split(", ")) == {
            "GET", "POST", "OPTIONS"
        }
        assert "content-type" in response.headers["access-control-allow-headers"].lower()
        assert "*" not in response.headers["access-control-allow-headers"]
        actual = client.get("/health", headers={"Origin": origin})
        assert actual.status_code == 200
        assert actual.json() == {"status": "ok"}
        for result in (response, actual):
            assert result.headers["access-control-allow-origin"] == origin
            assert "access-control-allow-credentials" not in result.headers
            assert "Origin" in result.headers["vary"]


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost:5174", "null"])
def test_disallowed_origin(monkeypatch, origin):
    client = make_client(monkeypatch, "http://localhost:5173")
    response = preflight(client, origin)
    assert response.status_code == 400
    actual = client.get("/health", headers={"Origin": origin})
    assert actual.status_code == 200
    for result in (response, actual):
        assert "access-control-allow-origin" not in result.headers
        assert "access-control-allow-credentials" not in result.headers


@pytest.mark.parametrize("origins", [None, "", " , "])
def test_default_disables_cors(monkeypatch, origins):
    client = make_client(monkeypatch, origins)
    assert preflight(client, "http://localhost:5173").status_code == 405
    for headers in ({}, {"Origin": "http://localhost:5173"}):
        response = client.get("/health", headers=headers)
        assert response.json() == {"status": "ok"}
        assert not any(key.startswith("access-control-") for key in response.headers)


@pytest.mark.parametrize("method,headers", [("DELETE", "Content-Type"), ("POST", "Authorization")])
def test_disallowed_method_or_header(monkeypatch, method, headers):
    client = make_client(monkeypatch, "http://localhost:5173")
    assert preflight(client, "http://localhost:5173", method, headers).status_code == 400


@pytest.mark.parametrize("origin", [
    "*", "https://*.example.com", "null", "ftp://example.com", "example.com",
    "https://example.com/", "https://example.com/path", "https://example.com?",
    "https://example.com#fragment", "https://user:pass@example.com",
    "https://example.com:70000", "https://example.com:0", "https://example.com:",
    "https://bad host", "https://example.com\\evil", "https://-bad.example",
    "http://[invalid]", "https://good.example,*",
])
def test_invalid_configuration_fails_startup(monkeypatch, origin):
    with pytest.raises(ValueError, match="CORS_ALLOWED_ORIGINS"):
        make_client(monkeypatch, origin)


def test_ipv6_origin(monkeypatch):
    client = make_client(monkeypatch, "http://[::1]:5173")
    assert preflight(client, "http://[::1]:5173").status_code == 200


@pytest.mark.parametrize("module,from_backend", [("backend.main", False), ("main", True)])
def test_import_modes_and_health_without_service_credentials(module, from_backend):
    backend = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ("CORS_ALLOWED_ORIGINS", "GEMINI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "MOCK_EXTERNAL_SERVICES"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-c", (
            f"from {module} import app; "
            "from fastapi.testclient import TestClient; "
            "r = TestClient(app).get('/health'); "
            "assert r.status_code == 200; assert r.json() == {'status': 'ok'}"
        )],
        cwd=backend if from_backend else backend.parent,
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
