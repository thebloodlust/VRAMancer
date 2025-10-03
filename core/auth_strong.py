"""Authentification forte (prototype production-ready minimal):

Fonctionnalités:
 - Comptes utilisateurs en mémoire (extensible persistence)
 - Hashage bcrypt (via passlib ou fallback pbkdf2) – ici pbkdf2_hmac standard lib
 - JWT (PyJWT) avec signature HS256, rotation de clé optionnelle
 - Refresh token (rotation simple, stockage mémoire)
 - Rôles (user, ops, admin)

Variables d'env:
  VRM_AUTH_SECRET : secret base pour JWT (OBLIGATOIRE en prod)
  VRM_AUTH_EXP    : durée access token (s, défaut 900)
  VRM_AUTH_REFRESH_EXP : durée refresh token (s, défaut 86400)

Extension future: stockage dans sqlite, lockout, MFA, audit.
"""
from __future__ import annotations
import os, time, hmac, hashlib, secrets
from dataclasses import dataclass
from typing import Dict, Optional
import jwt  # PyJWT

@dataclass
class User:
    username: str
    pwd_hash: str
    role: str = "user"
    salt: str = ""

_USERS: Dict[str, User] = {}
_REFRESH_STORE: Dict[str, dict] = {}  # refresh_token -> {sub, exp}

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 120_000).hex()

def create_user(username: str, password: str, role: str = "user"):
    if username in _USERS:
        raise ValueError("user exists")
    salt = secrets.token_hex(8)
    pwd_hash = _hash_password(password, salt)
    _USERS[username] = User(username, pwd_hash, role, salt)

def verify_user(username: str, password: str) -> bool:
    u = _USERS.get(username)
    if not u:
        return False
    cand = _hash_password(password, u.salt)
    return hmac.compare_digest(cand, u.pwd_hash)

def _get_secret() -> str:
    sec = os.environ.get('VRM_AUTH_SECRET')
    if not sec:
        # Générer un secret volatile (non-prod) si absent
        sec = os.environ.setdefault('VRM_AUTH_SECRET', secrets.token_hex(16))
    return sec

def issue_tokens(username: str) -> dict:
    u = _USERS[username]
    now = int(time.time())
    exp_access = now + int(os.environ.get('VRM_AUTH_EXP','900'))
    exp_refresh = now + int(os.environ.get('VRM_AUTH_REFRESH_EXP','86400'))
    payload = {"sub": username, "role": u.role, "iat": now, "exp": exp_access, "typ": "access"}
    secret = _get_secret()
    access = jwt.encode(payload, secret, algorithm='HS256')
    refresh_payload = {"sub": username, "iat": now, "exp": exp_refresh, "typ": "refresh"}
    refresh = jwt.encode(refresh_payload, secret, algorithm='HS256')
    _REFRESH_STORE[refresh] = {"sub": username, "exp": exp_refresh}
    return {"access": access, "refresh": refresh, "expires_in": exp_access-now}

def refresh_token(old_refresh: str) -> Optional[dict]:
    secret = _get_secret()
    try:
        data = jwt.decode(old_refresh, secret, algorithms=['HS256'])
        if data.get('typ') != 'refresh':
            return None
    except Exception:
        return None
    rec = _REFRESH_STORE.get(old_refresh)
    if not rec or rec['exp'] < time.time():
        return None
    # Rotating refresh: supprimer l'ancien
    del _REFRESH_STORE[old_refresh]
    return issue_tokens(data['sub'])

def decode_access(token: str) -> Optional[dict]:
    secret = _get_secret()
    try:
        data = jwt.decode(token, secret, algorithms=['HS256'])
        if data.get('typ') != 'access':
            return None
        return data
    except Exception:
        return None

def ensure_default_admin():  # auto bootstrap
    if not _USERS:
        create_user('admin','admin','admin')

__all__ = [
    'create_user','verify_user','issue_tokens','refresh_token','decode_access','ensure_default_admin'
]
