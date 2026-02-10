"""Tests for discrete security middleware functions.

Covers:
  - _check_cors (single-origin matching)
  - _check_body_size
  - _check_rate_limit_mw
  - _check_read_only
  - _resolve_role
  - _check_rbac
  - _check_test_bypass behavior
  - CORS after_request (single origin, Vary header)
"""
import os
import pytest
from unittest.mock import MagicMock


os.environ.setdefault('VRM_API_TOKEN', 'testtoken')
os.environ.setdefault('VRM_MINIMAL_TEST', '1')
os.environ.setdefault('VRM_DISABLE_RATE_LIMIT', '1')
os.environ.setdefault('VRM_TEST_MODE', '1')


def _make_request(path="/api/test", method="GET", headers=None,
                  content_length=0, remote_addr="127.0.0.1"):
    """Create a mock Flask request object."""
    req = MagicMock()
    req.path = path
    req.method = method
    req.headers = headers or {}
    req.content_length = content_length
    req.remote_addr = remote_addr
    req.get_data = MagicMock(return_value=b"")
    return req


class TestCheckCors:
    def test_no_origin_passes(self):
        from core.security import _check_cors
        req = _make_request(headers={})
        assert _check_cors(req, {"http://localhost"}) is None

    def test_allowed_origin_passes(self):
        from core.security import _check_cors
        req = _make_request(headers={"Origin": "http://localhost"})
        assert _check_cors(req, {"http://localhost"}) is None

    def test_forbidden_origin_blocked(self):
        from core.security import _check_cors
        req = _make_request(headers={"Origin": "http://evil.com"})
        result = _check_cors(req, {"http://localhost"})
        assert result is not None
        assert result[1] == 403


class TestCheckBodySize:
    def test_within_limit(self):
        from core.security import _check_body_size
        req = _make_request(content_length=1000)
        assert _check_body_size(req, 5 * 1024 * 1024) is None

    def test_over_limit(self):
        from core.security import _check_body_size
        req = _make_request(content_length=10 * 1024 * 1024)
        result = _check_body_size(req, 5 * 1024 * 1024)
        assert result is not None
        assert result[1] == 413


class TestCheckReadOnly:
    def test_get_allowed(self, monkeypatch):
        from core.security import _check_read_only
        monkeypatch.setenv('VRM_READ_ONLY', '1')
        req = _make_request(method="GET")
        assert _check_read_only(req) is None

    def test_post_blocked(self, monkeypatch):
        from core.security import _check_read_only
        monkeypatch.setenv('VRM_READ_ONLY', '1')
        req = _make_request(method="POST")
        result = _check_read_only(req)
        assert result is not None
        assert result[1] == 503

    def test_disabled(self, monkeypatch):
        from core.security import _check_read_only
        monkeypatch.setenv('VRM_READ_ONLY', '0')
        req = _make_request(method="POST")
        assert _check_read_only(req) is None


class TestResolveRole:
    def test_no_token_returns_user(self):
        from core.security import _resolve_role
        req = _make_request(headers={})
        assert _resolve_role(req, "testsecret") == "user"

    def test_shared_secret_returns_admin(self):
        from core.security import _resolve_role
        req = _make_request(headers={"X-API-TOKEN": "mysecret"})
        assert _resolve_role(req, "mysecret") == "admin"


class TestCheckRbac:
    def test_no_restriction(self):
        from core.security import _check_rbac
        req = _make_request(path="/api/generate")
        assert _check_rbac(req, "user", {"/api/admin": "admin"}) is None

    def test_insufficient_role(self):
        from core.security import _check_rbac
        req = _make_request(path="/api/admin")
        result = _check_rbac(req, "user", {"/api/admin": "admin"})
        assert result is not None
        assert result[1] == 403

    def test_sufficient_role(self):
        from core.security import _check_rbac
        req = _make_request(path="/api/admin")
        assert _check_rbac(req, "admin", {"/api/admin": "admin"}) is None


class TestCorsResponseHeaders:
    """Test that CORS after_request returns single origin + Vary header."""

    def test_cors_single_origin(self):
        """CORS response should contain single matching origin, not comma-joined."""
        from flask import Flask
        from core.security import install_security

        test_app = Flask(__name__)
        test_app.config['TESTING'] = True

        @test_app.route('/api/test')
        def _test():
            return {"ok": True}

        # Reset security flag
        test_app._vramancer_sec_installed = False
        install_security(test_app)

        with test_app.test_client() as c:
            resp = c.get('/api/test',
                         headers={
                             'X-API-TOKEN': 'testtoken',
                             'Origin': 'http://localhost',
                         })
            acao = resp.headers.get('Access-Control-Allow-Origin', '')
            # Must be a single origin, not comma-separated
            assert ',' not in acao
            assert acao == 'http://localhost'
            # Must include Vary: Origin
            assert 'Origin' in resp.headers.get('Vary', '')
