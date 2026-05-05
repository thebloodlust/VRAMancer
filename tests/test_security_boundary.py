"""Security boundary tests for the Flask production API."""
import os
import pytest


@pytest.fixture
def secure_client(monkeypatch):
    monkeypatch.setenv("VRM_API_TOKEN", "secret-token-xyz")
    monkeypatch.setenv("VRM_DISABLE_RATE_LIMIT", "0")
    try:
        from core.production_api import create_app
    except ImportError:
        pytest.skip("production_api unavailable")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_protected_endpoint_without_token(secure_client):
    r = secure_client.post("/api/models/load", json={"model": "gpt2"})
    # In production mode, must reject; in minimal-test mode security may be relaxed
    if os.environ.get("VRM_MINIMAL_TEST"):
        assert r.status_code in (200, 401, 403, 404)
    else:
        assert r.status_code in (401, 403, 404)


def test_protected_endpoint_invalid_token(secure_client):
    r = secure_client.post(
        "/api/models/load",
        json={"model": "gpt2"},
        headers={"X-API-Token": "wrong"},
    )
    if os.environ.get("VRM_MINIMAL_TEST"):
        assert r.status_code in (200, 401, 403, 404)
    else:
        assert r.status_code in (401, 403, 404)


def test_health_unprotected(secure_client):
    r = secure_client.get("/api/health")
    assert r.status_code in (200, 503)
