import os
import hashlib
import hmac
import ssl
import socket

# --- Authentification simple par clé partagée ---

def generate_node_key(secret):
    return hashlib.sha256(secret.encode()).hexdigest()

def authenticate_node(node_key, known_keys):
    """Timing-safe authentication against a list of known keys.

    Uses ``hmac.compare_digest`` so that the comparison time is constant
    regardless of which character differs — this prevents timing attacks.
    """
    return any(hmac.compare_digest(node_key, k) for k in known_keys)

# --- Chiffrement des transferts (TLS/SSL) ---

def secure_socket(sock, certfile=None, keyfile=None):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    if certfile and keyfile:
        context.load_cert_chain(certfile, keyfile)
    return context.wrap_socket(sock, server_side=True)

# --- Example usage ---
if __name__ == "__main__":
    import sys
    # Require secret from environment — never use a hardcoded default
    secret = os.environ.get("VRM_API_TOKEN")
    if not secret:
        print("ERROR: Set VRM_API_TOKEN environment variable before running.", file=sys.stderr)
        sys.exit(1)
    node_key = generate_node_key(secret)
    known_keys = [node_key]
    print("Authentifié :", authenticate_node(node_key, known_keys))

    # Chiffrement d’un socket (exemple)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        ssock = secure_socket(s)
        print("Socket sécurisé (TLS)")
    except Exception as e:
        print("Erreur SSL :", e)
