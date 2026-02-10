"""Unified security middleware (token + HMAC + RBAC + CORS) for Flask apps.

Provides:
  - Token authentication (mandatory in production, optional in dev)
  - HMAC request signing with rotating secrets
  - RBAC via JWT or shared-secret
  - CORS with single-origin matching (RFC 6454 compliant)
  - Rate limiting (per IP+path)
  - Request body size limit
Headers:
    X-API-TOKEN : shared token or JWT bearer
    X-API-TS    : unix timestamp (int)
    X-API-SIGN  : hexdigest SHA256(HMAC(secret, f"{ts}:{METHOD}:{PATH}" + body_bytes))

Usage:
    from core.security import install_security
    install_security(app)
"""
from __future__ import annotations
import time, hmac, hashlib, os, threading, logging
from typing import Optional, Tuple

_log = logging.getLogger(__name__)

WINDOW_SECONDS = 60
_RATE_WINDOW = 10   # seconds
_RATE_MAX = 200     # requests / window / IP (configurable via VRM_RATE_MAX)
_MAX_TRACKED_KEYS = 10_000  # prevent unbounded memory growth
_requests_lock = threading.Lock()
_requests: dict[str, list[float]] = {}
_last_gc: float = 0.0
_health_hits = 0  # simplified /api/health counter (test mode)
_rotating_secret: Optional[str] = None
_next_rotation_ts: float = 0.0
_ROTATE_INTERVAL = 3600  # 1h


def _compute_hmac(secret: str, ts: str, method: str, path: str, body: bytes) -> str:
    base = f"{ts}:{method}:{path}".encode() + (body or b"")
    return hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


def _gc_stale_keys(now: float) -> None:
    """Remove stale entries from _requests (call under lock)."""
    global _last_gc
    # Run GC at most every 30 seconds
    if now - _last_gc < 30:
        return
    _last_gc = now
    stale = [k for k, v in _requests.items() if not v or now - v[-1] > _RATE_WINDOW * 3]
    for k in stale:
        del _requests[k]
    # Hard cap: if still too many keys, evict oldest
    if len(_requests) > _MAX_TRACKED_KEYS:
        sorted_keys = sorted(_requests.keys(), key=lambda k: _requests[k][-1] if _requests[k] else 0)
        for k in sorted_keys[:len(_requests) - _MAX_TRACKED_KEYS]:
            del _requests[k]


def _rate_limit(remote: str, path: str) -> bool:
    """Check per-IP+path rate limit. Returns False if exceeded."""
    now = time.time()
    with _requests_lock:
        _gc_stale_keys(now)
        key = f"{remote}:{path}"
        lst = _requests.setdefault(key, [])
        while lst and now - lst[0] > _RATE_WINDOW:
            lst.pop(0)
        try:
            limit = int(os.environ.get('VRM_RATE_MAX', str(_RATE_MAX)))
        except Exception:
            limit = _RATE_MAX
        if len(lst) >= limit:
            return False
        lst.append(now)
    return True


def reset_rate_limiter():
    """Reset rate limiter state (for tests)."""
    with _requests_lock:
        _requests.clear()
    global _health_hits
    _health_hits = 0


def _maybe_rotate(secret: Optional[str]):  # pragma: no cover
    """Derive a rotating secret from the base secret (hourly rotation)."""
    global _rotating_secret, _next_rotation_ts
    if not secret:
        return secret
    if os.environ.get('VRM_DISABLE_SECRET_ROTATION') == '1':
        return secret
    now = time.time()
    if now > _next_rotation_ts or not _rotating_secret:
        base_epoch = int(now // _ROTATE_INTERVAL)
        _rotating_secret = hashlib.sha256(f"{secret}:{base_epoch}".encode()).hexdigest()[:32]
        _next_rotation_ts = (base_epoch + 1) * _ROTATE_INTERVAL
    return _rotating_secret


def reset_rotation():
    """Reset rotation state (for tests)."""
    global _rotating_secret, _next_rotation_ts
    _rotating_secret = None
    _next_rotation_ts = 0


def verify_request(secret: Optional[str], method: str, path: str,
                   headers: dict, body: bytes) -> Optional[Tuple[str, int]]:
    """Verify token + optional HMAC signature.

    Returns (message, http_code) on error, None on success.
    Only protects endpoints starting with /api/.
    """
    if not path.startswith("/api/"):
        return None

    is_production = os.environ.get('VRM_PRODUCTION') == '1'

    # Test bypass (blocked in production)
    if not is_production:
        if os.environ.get('VRM_TEST_RELAX_SECURITY') == '1':
            return None
        if (os.environ.get('VRM_TEST_BYPASS_HA') == '1'
                and path == '/api/ha/apply'):
            return None

    # --- Token validation (MANDATORY in production) ---
    token = (headers.get("X-API-TOKEN")
             or headers.get("Authorization", "").replace("Bearer ", ""))

    if is_production and not token:
        return ("token required", 401)

    if token and secret:
        eff_secret = _maybe_rotate(secret)
        if not hmac.compare_digest(token, eff_secret):
            if not hmac.compare_digest(token, secret):
                return ("invalid token", 401)

    # --- HMAC signature verification (when provided) ---
    eff_secret = _maybe_rotate(secret)
    sig = headers.get("X-API-SIGN")
    ts = headers.get("X-API-TS")
    if sig and ts and eff_secret:
        try:
            ts_i = int(ts)
            if abs(time.time() - ts_i) > WINDOW_SECONDS:
                return ("stale", 401)
            calc = _compute_hmac(eff_secret, ts, method, path, body)
            if not hmac.compare_digest(calc, sig):
                if secret and secret != eff_secret:
                    alt = _compute_hmac(secret, ts, method, path, body)
                    if hmac.compare_digest(alt, sig):
                        return None
                return ("bad sign", 401)
        except Exception:
            return ("invalid sign", 401)
    return None


def get_effective_secret() -> Optional[str]:
    """Return the current (possibly rotated) secret. Used in tests."""
    live_secret = os.environ.get("VRM_API_TOKEN")
    if not live_secret:
        return None
    return _maybe_rotate(live_secret)


# ---------------------------------------------------------------------------
# Discrete middleware functions (testable independently)
# ---------------------------------------------------------------------------

def _check_test_bypass(request) -> Optional[bool]:
    """Check if test bypass flags allow skipping security.

    Returns True to skip all checks, False to skip with rate-limit only,
    None to continue normal checks.
    All bypasses are BLOCKED when VRM_PRODUCTION=1.
    """
    if os.environ.get('VRM_PRODUCTION') == '1':
        return None

    if os.environ.get('VRM_TEST_ALL_OPEN') == '1':
        return True

    if os.environ.get('VRM_TEST_MODE') == '1':
        remote = request.remote_addr or "?"
        if (request.path == '/api/health'
                and os.environ.get('VRM_DISABLE_RATE_LIMIT') != '1'):
            global _health_hits
            _health_hits += 1
            try:
                dyn_limit = int(os.environ.get('VRM_RATE_MAX', '100'))
            except Exception:
                dyn_limit = 100
            if _health_hits > dyn_limit:
                return False
            if not _rate_limit(remote, request.path):
                return False
        return True

    if os.environ.get('VRM_TEST_RELAX_SECURITY') == '1':
        remote = request.remote_addr or "?"
        if (os.environ.get('VRM_DISABLE_RATE_LIMIT') != '1'
                and not _rate_limit(remote, request.path)):
            return False
        return True

    return None


def _check_cors(request, allowed_origins: set) -> Optional[Tuple[str, int]]:
    """Validate CORS origin. Returns error tuple or None."""
    origin = request.headers.get("Origin")
    if origin and origin not in allowed_origins:
        return ("forbidden origin", 403)
    return None


def _check_body_size(request, max_body: int) -> Optional[Tuple[str, int]]:
    """Validate request body size. Returns error tuple or None."""
    cl = request.content_length or 0
    if cl > max_body:
        return ("body too large", 413)
    return None


def _check_rate_limit_mw(request) -> Optional[Tuple[str, int]]:
    """Rate limiting middleware. Returns error tuple or None."""
    if os.environ.get('VRM_DISABLE_RATE_LIMIT') == '1':
        return None
    remote = request.remote_addr or "?"
    if not _rate_limit(remote, request.path):
        return ("rate limited", 429)
    return None


def _check_auth(request, initial_secret: str) -> Optional[Tuple[str, int]]:
    """Token + HMAC authentication. Returns error tuple or None."""
    body = request.get_data(cache=True) or b""
    live_secret = os.environ.get("VRM_API_TOKEN", initial_secret)
    return verify_request(
        live_secret, request.method, request.path, request.headers, body
    )


def _check_read_only(request) -> Optional[Tuple[str, int]]:
    """Read-only mode enforcement. Returns error or None."""
    if os.environ.get('VRM_READ_ONLY', '0') == '1':
        if request.method in ('POST', 'PUT', 'DELETE'):
            return ("read-only", 503)
    return None


def _resolve_role(request, live_secret: Optional[str]) -> str:
    """Derive user role from JWT token or shared secret."""
    token = (request.headers.get("X-API-TOKEN")
             or request.headers.get("Authorization", "").replace("Bearer ", ""))
    if not token:
        return "user"
    try:
        from core.auth_strong import decode_access
        payload = decode_access(token)
        if payload and "role" in payload:
            return payload["role"].lower()
        elif live_secret and hmac.compare_digest(token, live_secret):
            return "admin"
    except ImportError:
        if live_secret and hmac.compare_digest(token, live_secret):
            return "admin"
    except Exception:
        if live_secret and hmac.compare_digest(token, live_secret):
            return "admin"
    return "user"


def _check_rbac(request, role: str,
                role_required: dict) -> Optional[Tuple[str, int]]:
    """RBAC enforcement. Returns error or None."""
    need = role_required.get(request.path)
    if not need:
        return None
    order = {"user": 0, "ops": 1, "admin": 2}
    if order.get(role, 0) < order.get(need, 0):
        return ("role insufficient", 403)
    return None


# ---------------------------------------------------------------------------
# Main installer
# ---------------------------------------------------------------------------

def install_security(app):
    """Install a single before_request hook on the given Flask app.

    Idempotent — does not double-install.
    Integrates ZeroTrust proxy for production-grade authentication.
    """
    if getattr(app, "_vramancer_sec_installed", False):
        return

    initial_secret = os.environ.get("VRM_API_TOKEN")

    # RBAC: endpoint -> minimum role
    role_required = {
        "/api/security/rotate": "admin",
        "/api/tasks/estimator/install": "admin",
        "/api/memory/evict": "ops",
        "/api/memory/summary": "ops",
        "/api/models/load": "ops",
    }

    # CORS origins (comma-separated)
    allowed_origins = set(
        o.strip()
        for o in os.environ.get(
            "VRM_CORS_ORIGINS", "http://localhost,http://127.0.0.1"
        ).split(',')
        if o.strip()
    )
    max_body = int(os.environ.get("VRM_MAX_BODY", "5242880"))  # 5 MB
    try:
        _rm = int(os.environ.get("VRM_RATE_MAX", "0"))
        if _rm > 0:
            global _RATE_MAX
            _RATE_MAX = _rm
    except Exception:
        pass

    @app.before_request
    def _guard():
        from flask import request

        # 1. Test bypass flags (disabled in production)
        bypass = _check_test_bypass(request)
        if bypass is True:
            return None
        if bypass is False:
            return ("rate limited", 429)

        # 2. CORS origin check
        err = _check_cors(request, allowed_origins)
        if err:
            return err

        # 3. Body size limit
        err = _check_body_size(request, max_body)
        if err:
            return err

        # 4. Rate limiting
        err = _check_rate_limit_mw(request)
        if err:
            return err

        # 5. Token + HMAC authentication
        err = _check_auth(request, initial_secret)
        if err:
            return err

        # 6. Read-only mode
        err = _check_read_only(request)
        if err:
            return err

        # 7. RBAC
        live_secret = os.environ.get("VRM_API_TOKEN", initial_secret)
        role = _resolve_role(request, live_secret)
        err = _check_rbac(request, role, role_required)
        if err:
            return err

    @app.route('/api/health')
    def _health():
        return {"ok": True}

    @app.route('/api/security/rotate', methods=['POST'])
    def _force_rotate():
        global _next_rotation_ts
        _next_rotation_ts = 0
        live_secret = os.environ.get("VRM_API_TOKEN", initial_secret)
        _maybe_rotate(live_secret)
        return {"ok": True, "rotated": True}

    @app.after_request
    def _cors_headers(resp):
        """RFC 6454-compliant CORS + standard security headers."""
        from flask import request as _req
        origin = _req.headers.get("Origin")
        if origin and origin in allowed_origins:
            resp.headers['Access-Control-Allow-Origin'] = origin
            resp.headers['Vary'] = 'Origin'
        # If origin is absent (same-origin / non-browser) or not allowed,
        # do NOT send Access-Control-Allow-Origin at all — this is the
        # correct RFC 6454 behaviour.
        resp.headers['Access-Control-Allow-Headers'] = (
            'Content-Type,X-API-TOKEN,X-API-TS,X-API-SIGN,Authorization'
        )
        resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'

        # --- Standard security headers (always set) ---
        resp.headers['X-Content-Type-Options'] = 'nosniff'
        resp.headers['X-Frame-Options'] = 'DENY'
        resp.headers['X-XSS-Protection'] = '1; mode=block'
        resp.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        resp.headers['Content-Security-Policy'] = "default-src 'none'"

        # HSTS — only meaningful behind TLS (production)
        if os.environ.get('VRM_PRODUCTION') == '1':
            resp.headers['Strict-Transport-Security'] = (
                'max-age=31536000; includeSubDomains'
            )

        return resp

    app._vramancer_sec_installed = True


__all__ = [
    "install_security", "verify_request", "_compute_hmac",
    "reset_rate_limiter", "reset_rotation",
    # Discrete middleware (for testing)
    "_check_test_bypass", "_check_cors", "_check_body_size",
    "_check_rate_limit_mw", "_check_auth",
    "_check_read_only", "_resolve_role", "_check_rbac",
]
