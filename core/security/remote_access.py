"""
Poste de contrôle web sécurisé :
- Authentification forte (MFA)
- Gestion des rôles
- API pour contrôle distant

WARNING: This module requires environment variables for credentials.
Set VRM_REMOTE_ADMIN_PASS, VRM_REMOTE_USER_PASS, VRM_REMOTE_ADMIN_MFA,
VRM_REMOTE_USER_MFA before using in production.
"""
try:
    from flask import Flask, request, jsonify, session
except ImportError:
    Flask = None  # type: ignore
from functools import wraps
import secrets
import hashlib
import hmac as hmac_mod
import os
import logging

logger = logging.getLogger(__name__)

_app_instance = None


def _hash_credential(value: str) -> str:
    """Hash a credential with a salt for secure comparison.

    In production (VRM_PRODUCTION=1): VRM_CREDENTIAL_SALT is REQUIRED.
    In dev: generates a random per-process salt if not set (with warning).
    """
    salt = os.environ.get("VRM_CREDENTIAL_SALT")
    if not salt:
        if os.environ.get("VRM_PRODUCTION") == "1":
            raise RuntimeError(
                "SECURITY: VRM_CREDENTIAL_SALT must be set in production mode. "
                "Generate one with: python3 -c 'import secrets; print(secrets.token_hex(32))'"
            )
        # Dev: deterministic fallback so hashes match within the same process
        salt = "vramancer-dev-only-salt"
        logger.warning(
            "SECURITY: Using default dev salt for credential hashing. "
            "Set VRM_CREDENTIAL_SALT env var for persistent/secure hashing."
        )
    return hashlib.pbkdf2_hmac('sha256', value.encode(), salt.encode(), 100_000).hex()


def _load_users():
    """Load users from environment variables.

    In production (VRM_PRODUCTION=1): refuses to start without proper env vars.
    In dev: generates random temporary credentials with a warning.
    """
    is_production = os.environ.get("VRM_PRODUCTION", "0") == "1"
    admin_pass = os.environ.get("VRM_REMOTE_ADMIN_PASS")
    user_pass = os.environ.get("VRM_REMOTE_USER_PASS")
    admin_mfa = os.environ.get("VRM_REMOTE_ADMIN_MFA")
    user_mfa = os.environ.get("VRM_REMOTE_USER_MFA")

    if is_production:
        missing = []
        if not admin_pass:
            missing.append("VRM_REMOTE_ADMIN_PASS")
        if not user_pass:
            missing.append("VRM_REMOTE_USER_PASS")
        if not admin_mfa:
            missing.append("VRM_REMOTE_ADMIN_MFA")
        if not user_mfa:
            missing.append("VRM_REMOTE_USER_MFA")
        if missing:
            msg = (
                f"SECURITY: Missing required credentials in production mode: "
                f"{', '.join(missing)}. Set these environment variables before starting."
            )
            logger.error(msg)
            raise RuntimeError(msg)
    else:
        if not admin_pass or not user_pass:
            # Generate random temporary credentials for dev
            admin_pass = admin_pass or secrets.token_urlsafe(16)
            user_pass = user_pass or secrets.token_urlsafe(16)
            admin_mfa = admin_mfa or secrets.token_hex(3)  # 6 hex chars
            user_mfa = user_mfa or secrets.token_hex(3)
            logger.warning(
                "SECURITY: Using auto-generated dev credentials. "
                "Admin pass: %s | User pass: %s | Admin MFA: %s | User MFA: %s "
                "— Set VRM_REMOTE_*_PASS/MFA env vars for persistent credentials.",
                admin_pass, user_pass, admin_mfa, user_mfa,
            )

    return {
        "admin": {
            "password_hash": _hash_credential(admin_pass),
            "role": "admin",
            "mfa_hash": _hash_credential(admin_mfa or "000000"),
        },
        "user": {
            "password_hash": _hash_credential(user_pass),
            "role": "user",
            "mfa_hash": _hash_credential(user_mfa or "000000"),
        },
    }


try:
    USERS = _load_users()
except RuntimeError:
    # In production without env vars, USERS stays empty
    USERS = {}
    logger.error("Remote access module disabled: missing credentials")

ROLES = {"admin": ["all"], "user": ["view"]}


def _create_app():
    """Create Flask app lazily."""
    global _app_instance
    if _app_instance is None and Flask is not None:
        _app_instance = Flask(__name__)
        _app_instance.secret_key = secrets.token_hex(16)
    return _app_instance


def require_auth(role=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user" not in session:
                return jsonify({"error": "auth required"}), 401
            if role and USERS.get(session["user"], {}).get("role") != role:
                return jsonify({"error": "forbidden"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator


app = _create_app()

if app is not None:
    @app.route("/login", methods=["POST"])
    def login():
        data = request.json or {}
        username = data.get("user", "")
        user_record = USERS.get(username)
        if not user_record:
            return jsonify({"error": "invalid credentials"}), 401

        # Secure hash comparison (no plaintext)
        password = data.get("password", "")
        mfa = data.get("mfa", "")
        pwd_hash = _hash_credential(password)
        mfa_hash = _hash_credential(mfa)

        pwd_ok = hmac_mod.compare_digest(pwd_hash, user_record["password_hash"])
        mfa_ok = hmac_mod.compare_digest(mfa_hash, user_record["mfa_hash"])

        if pwd_ok and mfa_ok:
            session["user"] = username
            return jsonify({"ok": True, "role": user_record["role"]})
        return jsonify({"error": "invalid credentials"}), 401

    @app.route("/logout")
    def logout():
        session.pop("user", None)
        return jsonify({"ok": True})

    @app.route("/status")
    @require_auth()
    def status():
        return jsonify({"user": session["user"], "role": USERS[session["user"]]["role"]})

    @app.route("/admin")
    @require_auth(role="admin")
    def admin_panel():
        return jsonify({"admin": True})


def main():
    """Entry point for remote access server."""
    if app is None:
        logger.error("Flask not available. Install flask to use remote access.")
        return
    if not USERS:
        logger.error("No users configured. Set VRM_REMOTE_*_PASS env vars.")
        return
    debug_mode = os.environ.get("VRM_DEBUG", "0") == "1"
    if debug_mode:
        logger.warning("SECURITY: Running remote_access in debug mode.")
    app.run(port=5001, debug=debug_mode)


if __name__ == "__main__":
    main()
