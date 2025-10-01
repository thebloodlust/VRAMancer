import os
import hashlib
import ssl
import socket

# --- Authentification simple par clé partagée ---

def generate_node_key(secret):
    return hashlib.sha256(secret.encode()).hexdigest()

def authenticate_node(node_key, known_keys):
    return node_key in known_keys

# --- Chiffrement des transferts (TLS/SSL) ---

def secure_socket(sock, certfile=None, keyfile=None):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    if certfile and keyfile:
        context.load_cert_chain(certfile, keyfile)
    return context.wrap_socket(sock, server_side=True)

# --- Exemple d’utilisation ---
if __name__ == "__main__":
    # Génération et vérification de clé
    secret = "vramancer_secret"
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
