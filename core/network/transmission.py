import socketio
import json
from .packets import Packet
from utils.helpers import serialize_tensors, deserialize_tensors

sio = socketio.Client()

# ------------------------------------------------------------------
# 1️⃣  Connexion au serveur (ou à une autre machine)  
# ------------------------------------------------------------------
@sio.event
def connect():
    print("[Transmission] Connexion établie")

@sio.event
def disconnect():
    print("[Transmission] Déconnexion")

# ------------------------------------------------------------------
# 2️⃣  Envoi d’un bloc (tensors) → serveur / autre GPU
# ------------------------------------------------------------------
def send_block(tensors, shapes, dtypes, target_device="localhost"):
    """
    Exemple d’envoi d’un bloc de poids (ou d’un batch de calcul).
    Le `target_device` est le nom d’une machine dans votre cluster.
    """
    payload = serialize_tensors(tensors)
    packet  = Packet(payload)
    sio.emit("vramancer_packet", packet.pack(), namespace="/vram")
    print(f"[Transmission] Envoyé {len(tensors)} tensors vers {target_device}")

# ------------------------------------------------------------------
# 3️⃣  Réception d’un bloc
# ------------------------------------------------------------------
@sio.on("vramancer_packet", namespace="/vram")
def on_packet(data):
    packet = Packet.unpack(data)[0]
    # Pour l’exemple, on ne fait rien ici – le code client le fera
    print("[Transmission] Réception d’un paquet")

# ------------------------------------------------------------------
# 4️⃣  Démarrage du client (déjà fait dans vramancer_link)
# ------------------------------------------------------------------
def start_client(server_url="http://localhost:5000"):
    sio.connect(server_url)
