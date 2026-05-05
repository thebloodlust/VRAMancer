"""Tests for core.security.startup_checks.enforce_startup_checks."""
import os
import pytest


@pytest.fixture(autouse=True)
def _clean_prod_env(monkeypatch):
    for key in ("VRM_PRODUCTION", "VRM_API_TOKEN", "VRM_AUTH_SECRET",
                 "VRM_MINIMAL_TEST", "VRM_TEST_RELAX_SECURITY", "VRM_TEST_BYPASS_HA",
                 "VRM_DISABLE_RATE_LIMIT", "VRM_TEST_MODE"):
        monkeypatch.delenv(key, raising=False)
    yield


def _enforce():
    # Reload to avoid module-level caching
    import importlib
    import core.security.startup_checks as sc
    importlib.reload(sc)
    sc.enforce_startup_checks()


# ── Non-production: no errors ─────────────────────────────────────────────────

def test_non_production_no_error():
    """In non-production mode, enforce_startup_checks never raises."""
    _enforce()  # should not raise


# ── Production mode checks ────────────────────────────────────────────────────

def test_production_requires_api_token(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    with pytest.raises(RuntimeError, match="VRM_API_TOKEN"):
        _enforce()


def test_production_requires_auth_secret(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "safetoken123")
    with pytest.raises(RuntimeError, match="VRM_AUTH_SECRET"):
        _enforce()


def test_production_rejects_minimal_test(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "safetoken123")
    monkeypatch.setenv("VRM_AUTH_SECRET", "safesecret123")
    monkeypatch.setenv("VRM_MINIMAL_TEST", "1")
    with pytest.raises(RuntimeError):
        _enforce()


def test_production_rejects_relax_security(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "safetoken123")
    monkeypatch.setenv("VRM_AUTH_SECRET", "safesecret123")
    monkeypatch.setenv("VRM_TEST_RELAX_SECURITY", "1")
    with pytest.raises(RuntimeError):
        _enforce()


def test_production_rejects_bypass_ha(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "safetoken123")
    monkeypatch.setenv("VRM_AUTH_SECRET", "safesecret123")
    monkeypatch.setenv("VRM_TEST_BYPASS_HA", "1")
    with pytest.raises(RuntimeError):
        _enforce()


def test_production_all_correct_passes(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "safetoken123")
    monkeypatch.setenv("VRM_AUTH_SECRET", "safesecret123")
    _enforce()  # should not raise
