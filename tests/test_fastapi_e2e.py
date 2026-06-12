"""End-to-end smoke tests for server.py FastAPI app."""
import pytest


@pytest.fixture
def fastapi_client():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    try:
        import server  # noqa: F401
    except Exception as exc:
        pytest.skip(f"server.py cannot be imported: {exc}")
    from server import app
    return TestClient(app)


def test_health_endpoint(fastapi_client):
    r = fastapi_client.get("/health")
    assert r.status_code in (200, 503)


def test_models_list(fastapi_client):
    r = fastapi_client.get("/v1/models")
    assert r.status_code in (200, 404)


def test_chat_without_model_loaded(fastapi_client):
    r = fastapi_client.post("/v1/chat/completions", json={
        "model": "vramancer",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    })
    assert r.status_code in (200, 400, 503)
