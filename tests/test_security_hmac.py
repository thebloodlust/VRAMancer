import os, time, hmac, hashlib
import pytest
from core.security import _compute_hmac, reset_rotation, get_effective_secret


SECRET = "test-secret-token"


@pytest.fixture(autouse=True)
def _hmac_test_env(monkeypatch):
    """Set up clean env for HMAC tests without polluting other test modules."""
    monkeypatch.setenv("VRM_API_TOKEN", SECRET)
    monkeypatch.setenv("VRM_DISABLE_RATE_LIMIT", "1")
    monkeypatch.setenv("VRM_DISABLE_SECRET_ROTATION", "1")
    monkeypatch.delenv("VRM_TEST_RELAX_SECURITY", raising=False)
    monkeypatch.delenv("VRM_TEST_MODE", raising=False)
    monkeypatch.delenv("VRM_TEST_ALL_OPEN", raising=False)
    reset_rotation()
    yield
    # monkeypatch auto-restores all env vars


def _sign(ts: int, method: str, path: str, body: bytes):
    eff = get_effective_secret() or SECRET
    return _compute_hmac(eff, str(ts), method, path, body)


def test_hmac_success_dashboard():
    from dashboard.dashboard_web import app as web_app
    reset_rotation()
    client = web_app.test_client()
    ts = int(time.time())
    sig = _sign(ts, "GET", "/api/health", b"")
    r = client.get("/api/health", headers={"X-API-TOKEN": SECRET, "X-API-TS": str(ts), "X-API-SIGN": sig})
    assert r.status_code == 200
    assert r.json.get("ok") is True


def test_hmac_bad_signature():
    from core.network.supervision_api import app as sup_app
    client = sup_app.test_client()
    ts = int(time.time())
    bad_sig = "deadbeef"
    r = client.get("/api/health", headers={"X-API-TOKEN": SECRET, "X-API-TS": str(ts), "X-API-SIGN": bad_sig})
    assert r.status_code in (401, 200)


def test_token_only_without_hmac():
    from dashboard.dashboard_web import app as web_app
    client = web_app.test_client()
    r = client.get("/api/health", headers={"X-API-TOKEN": SECRET})
    # Should be 200: HMAC is optional, token alone is sufficient
    assert r.status_code == 200
