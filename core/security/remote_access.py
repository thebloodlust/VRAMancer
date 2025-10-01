"""
Poste de contrôle web sécurisé :
- Authentification forte (MFA)
- Gestion des rôles
- API pour contrôle distant
"""
from flask import Flask, request, jsonify, session
from functools import wraps
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

USERS = {
    "admin": {"password": "adminpass", "role": "admin", "mfa": "123456"},
    "user": {"password": "userpass", "role": "user", "mfa": "654321"},
}

ROLES = {"admin": ["all"], "user": ["view"]}

def require_auth(role=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user" not in session:
                return jsonify({"error": "auth required"}), 401
            if role and USERS[session["user"]]["role"] != role:
                return jsonify({"error": "forbidden"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = USERS.get(data.get("user"))
    if user and user["password"] == data.get("password") and user["mfa"] == data.get("mfa"):
        session["user"] = data["user"]
        return jsonify({"ok": True, "role": user["role"]})
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

if __name__ == "__main__":
    app.run(port=5001, debug=True)
