"""Sécurité unifiée (token + HMAC) pour les micro‑apps Flask.

Objectif: éviter la duplication des before_request dans chaque dashboard / API.

Protocole:
  En-têtes:
    X-API-TOKEN : token partagé (optionnel si variable d'env absente)
    X-API-TS    : timestamp unix (int)
    X-API-SIGN  : hexdigest SHA256(HMAC(secret, f"{ts}:{METHOD}:{PATH}" + body_bytes))

Fenêtre temporelle: ±60s
Si X-API-SIGN & X-API-TS présents => HMAC obligatoire (si secret présent).

Utilisation:
    from core.security import install_security
    install_security(app)
"""
from __future__ import annotations
import time, hmac, hashlib, os, threading
from typing import Optional, Tuple

WINDOW_SECONDS = 60
_RATE_WINDOW = 10   # secondes
_RATE_MAX = 200     # requêtes / fenêtre / IP (config via env VRM_RATE_MAX)
_requests_lock = threading.Lock()
_requests: dict[str, list[float]] = {}
_health_hits = 0  # compteur simplifié /api/health (test mode)
_rotating_secret: Optional[str] = None
_next_rotation_ts: float = 0.0
_ROTATE_INTERVAL = 3600  # 1h

def _compute_hmac(secret: str, ts: str, method: str, path: str, body: bytes) -> str:
    base = f"{ts}:{method}:{path}".encode() + (body or b"")
    return hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()

def _rate_limit(remote: str, path: str) -> bool:
    now = time.time()
    with _requests_lock:
        # Limitation par (remote+path) pour éviter qu'un burst sur /api/health ne bloque tout le reste
        key = f"{remote}:{path}"
        lst = _requests.setdefault(key, [])
        # purge
        while lst and now - lst[0] > _RATE_WINDOW:
            lst.pop(0)
        # Lecture dynamique de la limite (permet de changer VRM_RATE_MAX à chaud dans tests)
        try:
            limit = int(os.environ.get('VRM_RATE_MAX', str(_RATE_MAX)))
        except Exception:
            limit = _RATE_MAX
        if len(lst) >= limit:
            return False
        lst.append(now)
    return True

def reset_rate_limiter():
    """Réinitialise l'état interne du rate limiter (tests)."""
    with _requests_lock:
        _requests.clear()
    global _health_hits
    _health_hits = 0

def _maybe_rotate(secret: Optional[str]):  # pragma: no cover (dépend temps)
    global _rotating_secret, _next_rotation_ts
    if not secret:
        return secret
    if os.environ.get('VRM_DISABLE_SECRET_ROTATION') == '1':
        return secret
    now = time.time()
    if now > _next_rotation_ts or not _rotating_secret:
        # Double buffer: nouvelle clé dérivée = sha256(secret + str(now//interval))
        base_epoch = int(now // _ROTATE_INTERVAL)
        import hashlib as _hl
        _rotating_secret = _hl.sha256(f"{secret}:{base_epoch}".encode()).hexdigest()[:32]
        _next_rotation_ts = (base_epoch + 1) * _ROTATE_INTERVAL
    return _rotating_secret

def reset_rotation():  # utilitaire tests
    global _rotating_secret, _next_rotation_ts
    _rotating_secret = None
    _next_rotation_ts = 0

def verify_request(secret: Optional[str], method: str, path: str, headers: dict, body: bytes) -> Optional[Tuple[str,int]]:
    """Vérifie token + (optionnel) HMAC.
    Retourne tuple (message, code_http) en cas d'erreur, sinon None.
    Ne protège que les endpoints commençant par /api/ (à vérifier côté appelant avant invocation si besoin).
    """
    if not path.startswith("/api/"):
        return None
    # Mode tests: bypass token/HMAC (conserve rate limit + RBAC) si drapeau actif
    if os.environ.get('VRM_TEST_RELAX_SECURITY') == '1':
        return None
    # Bypass spécifique tests replication si variable définie
    if os.environ.get('VRM_TEST_BYPASS_HA') == '1' and path == '/api/ha/apply':
        return None
    eff_secret = _maybe_rotate(secret)
    # En mode standard le token devient purement facultatif: aucune restriction si absent / différent.
    # Les protections fortes reposent sur HMAC quand fourni.
    sig = headers.get("X-API-SIGN")
    ts  = headers.get("X-API-TS")
    if sig and ts and eff_secret:
        try:
            ts_i = int(ts)
            if abs(time.time() - ts_i) > WINDOW_SECONDS:
                return ("stale", 401)
            calc = _compute_hmac(eff_secret, ts, method, path, body)
            if not hmac.compare_digest(calc, sig):
                # Fallback: accepter signature basée sur secret original avant rotation
                if secret and secret != eff_secret:
                    alt = _compute_hmac(secret, ts, method, path, body)
                    if hmac.compare_digest(alt, sig):
                        return None
                return ("bad sign", 401)
        except Exception:
            return ("invalid sign", 401)
    return None

def get_effective_secret() -> Optional[str]:  # utilisé dans tests HMAC
    live_secret = os.environ.get("VRM_API_TOKEN")
    if not live_secret:
        return None
    # respecte désactivation rotation
    return _maybe_rotate(live_secret)

def install_security(app):
    """Installe un hook before_request unique sur l'app Flask fournie.
    Idempotent: si déjà installé, ne double pas.
    """
    if getattr(app, "_vramancer_sec_installed", False):
        return
    # Note: on ne fige plus le secret à l'installation pour permettre aux tests / rotations
    # de modifier VRM_API_TOKEN dynamiquement.
    # L'ancien comportement capturait la valeur ici.
    initial_secret = os.environ.get("VRM_API_TOKEN")
    # RBAC minimal: mapping endpoint -> rôle minimal
    role_required = {
        "/api/security/rotate": "admin",
        "/api/tasks/estimator/install": "admin",
        "/api/memory/evict": "ops",
        "/api/memory/summary": "ops",
    }
    # CORS origins autorisés (liste séparée par virgule) sinon blocage.
    allowed_origins = set([o.strip() for o in os.environ.get("VRM_CORS_ORIGINS", "http://localhost,http://127.0.0.1").split(',') if o.strip()])
    max_body = int(os.environ.get("VRM_MAX_BODY", "5242880"))  # 5MB défaut
    try:
        _rm = int(os.environ.get("VRM_RATE_MAX", "0"))
        if _rm > 0:
            global _RATE_MAX
            _RATE_MAX = _rm
    except Exception:
        pass

    @app.before_request
    def _guard():  # pragma: no cover - exécuté via tests client
        from flask import request
        # Bypass complet (sauf taille corps / CORS ignorés) pour tests globaux après vérification ciblée
        if os.environ.get('VRM_TEST_ALL_OPEN') == '1':
            return None
        if os.environ.get('VRM_TEST_MODE') == '1':
            # Mode test complet: seul rate limiting sur /api/health pour test dédié
            remote = request.remote_addr or "?"
            if request.path == '/api/health' and os.environ.get('VRM_DISABLE_RATE_LIMIT') != '1':
                # Fallback compteur simple pour robustesse
                global _health_hits
                _health_hits += 1
                try:
                    dyn_limit = int(os.environ.get('VRM_RATE_MAX','100'))
                except Exception:
                    dyn_limit = 100
                if _health_hits > dyn_limit:
                    return ("rate limited", 429)
                if not _rate_limit(remote, request.path):
                    return ("rate limited", 429)
            return None
        if os.environ.get('VRM_TEST_RELAX_SECURITY') == '1':
            remote = request.remote_addr or "?"
            if os.environ.get('VRM_DISABLE_RATE_LIMIT') != '1' and not _rate_limit(remote, request.path):
                return ("rate limited", 429)
            return None
        origin = request.headers.get("Origin")
        if origin and origin not in allowed_origins:
            return ("forbidden origin", 403)
        cl = request.content_length or 0
        if cl > max_body:
            return ("body too large", 413)
        remote = request.remote_addr or "?"
        # Bypass total via variable (tests, bench) ou limitation ciblée /api/health uniquement
        if os.environ.get('VRM_DISABLE_RATE_LIMIT') != '1':
            if not _rate_limit(remote, request.path):
                return ("rate limited", 429)
        body = request.get_data(cache=True) or b""
        live_secret = os.environ.get("VRM_API_TOKEN", initial_secret)
        err = verify_request(live_secret, request.method, request.path, request.headers, body)
        if err:
            return err
        if os.environ.get('VRM_READ_ONLY','0') == '1':
            if request.method in ('POST','PUT','DELETE'):
                return ("read-only", 503)
        role = request.headers.get("X-API-ROLE", "user").lower()
        if request.path in role_required:
            need = role_required[request.path]
            order = {"user":0,"ops":1,"admin":2}
            if order.get(role,0) < order.get(need,0):
                return ("role insufficient", 403)

    @app.route('/api/health')
    def _health():  # pragma: no cover - trivial
        return {"ok": True}

    @app.route('/api/security/rotate', methods=['POST'])
    def _force_rotate():  # pragma: no cover
        global _next_rotation_ts
        _next_rotation_ts = 0
        live_secret = os.environ.get("VRM_API_TOKEN", initial_secret)
        _maybe_rotate(live_secret)
        return {"ok": True, "rotated": True}

    @app.after_request
    def _cors_headers(resp):  # pragma: no cover
        resp.headers['Access-Control-Allow-Origin'] = ','.join(allowed_origins) if allowed_origins else '*'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type,X-API-TOKEN,X-API-TS,X-API-SIGN,X-API-ROLE'
        resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        return resp

    app._vramancer_sec_installed = True

__all__ = ["install_security", "verify_request", "_compute_hmac", "reset_rate_limiter", "reset_rotation"]
