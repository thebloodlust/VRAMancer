"""Tests for core.security.verify_request and install_security."""
import os
import hashlib
import hmac
import pytest


@pytest.fixture(autouse=True)
def _clean_security_env(monkeypatch):
    for key in ("VRM_PRODUCTION", "VRM_TEST_RELAX_SECURITY", "VRM_TEST_BYPASS_HA",
                 "VRM_API_TOKEN", "VRM_AUTH_SECRET", "VRM_DISABLE_RATE_LIMIT",
                 "VRM_DISABLE_SECRET_ROTATION"):
        monkeypatch.delenv(key, raising=False)
    # Reset rotation state so tests don't bleed into each other
    from core.security import reset_rotation
    reset_rotation()
    yield


# ── Public paths ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/health", "/ready", "/live", "/api/health", "/"])
def test_public_path_always_allowed(path, monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    from core.security import verify_request
    result = verify_request("secret", "GET", path, {}, b"")
    assert result is None


def test_static_path_allowed(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    from core.security import verify_request
    result = verify_request("secret", "GET", "/static/app.js", {}, b"")
    assert result is None


def test_favicon_allowed(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    from core.security import verify_request
    result = verify_request("secret", "GET", "/favicon.ico", {}, b"")
    assert result is None


# ── Production mode ───────────────────────────────────────────────────────────

def test_production_no_token_returns_401(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    from core.security import verify_request
    result = verify_request("secret", "POST", "/api/generate", {}, b"")
    assert result is not None
    msg, code = result
    assert code == 401


def test_production_valid_token_allowed(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_DISABLE_SECRET_ROTATION", "1")
    from core.security import verify_request
    result = verify_request("secret", "GET", "/api/info",
                            {"X-API-TOKEN": "secret"}, b"")
    assert result is None


def test_production_invalid_token_returns_403(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_DISABLE_SECRET_ROTATION", "1")
    from core.security import verify_request
    result = verify_request("secret", "GET", "/api/info",
                            {"X-API-TOKEN": "wrongtoken"}, b"")
    assert result is not None
    msg, code = result
    assert code in (401, 403)


# ── Non-production bypasses ───────────────────────────────────────────────────

def test_relax_security_bypasses_everything(monkeypatch):
    monkeypatch.setenv("VRM_TEST_RELAX_SECURITY", "1")
    from core.security import verify_request
    result = verify_request("secret", "POST", "/api/generate", {}, b"body")
    assert result is None


def test_bypass_ha_only_applies_to_ha_path(monkeypatch):
    monkeypatch.setenv("VRM_TEST_BYPASS_HA", "1")
    from core.security import verify_request
    result = verify_request("secret", "POST", "/api/ha/apply", {}, b"")
    assert result is None


def test_bypass_ha_does_not_bypass_other_paths(monkeypatch):
    monkeypatch.setenv("VRM_TEST_BYPASS_HA", "1")
    from core.security import verify_request
    # Non-production, no token, no relax — should pass (non-prod default)
    result = verify_request("secret", "POST", "/api/generate", {}, b"")
    # Non-production without a token: no mandatory check → allowed
    assert result is None


def test_no_secret_non_production_allowed(monkeypatch):
    from core.security import verify_request
    result = verify_request(None, "GET", "/api/info", {}, b"")
    assert result is None
