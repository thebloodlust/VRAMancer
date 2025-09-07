from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import os

# ------------------------------------------------------------------
# 1️⃣  Génération / chargement de la clé RSA (public / privé)
# ------------------------------------------------------------------
def generate_keys(key_dir="keys"):
    os.makedirs(key_dir, exist_ok=True)
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key  = key.publickey().export_key()

    with open(os.path.join(key_dir, "private.pem"), "wb") as f:
        f.write(private_key)
    with open(os.path.join(key_dir, "public.pem"), "wb") as f:
        f.write(public_key)
    print("[SecureLayer] Clés RSA générées")

def load_private_key(key_path="keys/private.pem"):
    with open(key_path, "rb") as f:
        return RSA.import_key(f.read())

def load_public_key(key_path="keys/public.pem"):
    with open(key_path, "rb") as f:
        return RSA.import_key(f.read())

# ------------------------------------------------------------------
# 2️⃣  Chiffrement / déchiffrement des paquets
# ------------------------------------------------------------------
def encrypt(data, public_key):
    cipher = PKCS1_OAEP.new(public_key)
    return cipher.encrypt(data)

def decrypt(ciphertext, private_key):
    cipher = PKCS1_OAEP.new(private_key)
    return cipher.decrypt(ciphertext)
