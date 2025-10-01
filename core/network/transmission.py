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
import zlib

def send_block(tensors, shapes, dtypes, target_device="localhost", storage_path=None, compress=True, protocol="socketio", usb4_path=None):
    """
    Envoi d'un bloc de poids (ou batch) :
    - target_device : nom/IP de la machine cible
    - storage_path : chemin NAS/NVMe partagé (optionnel)
    - compress : active la compression zlib
    - protocol : "socketio", "tcp", "udp", "sfp", "rdma", "usb4" (custom)
    - usb4_path : chemin de montage USB4 (optionnel)
    Si storage_path ou usb4_path est fourni, le bloc est écrit sur le stockage partagé ou USB4
    au lieu d'être envoyé par le réseau.
    """
    payload = serialize_tensors(tensors)
    packet  = Packet(payload)
    data = packet.pack()
    if compress:
        data = zlib.compress(data)
    if usb4_path:
        # Routage mémoire intermachine via USB4 (déport VRAM)
        import os
        fname = f"block_{target_device}_usb4.bin"
        full_path = os.path.join(usb4_path, fname)
        with open(full_path, "wb") as f:
            f.write(data)
        print(f"[Transmission] Bloc VRAM déporté via USB4 sur {full_path} (plug-and-play IA distribuée, latence réduite)")
    elif storage_path:
        # Routage via NAS/NVMe partagé
        import os
        fname = f"block_{target_device}.bin"
        full_path = os.path.join(storage_path, fname)
        with open(full_path, "wb") as f:
            f.write(data)
        print(f"[Transmission] Bloc compressé écrit sur {full_path}")
    else:
        if protocol == "socketio":
            sio.emit("vramancer_packet", data, namespace="/vram")
            print(f"[Transmission] Bloc compressé envoyé via SocketIO vers {target_device}")
        elif protocol == "tcp":
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((target_device, 12345))
            s.sendall(data)
            s.close()
            print(f"[Transmission] Bloc compressé envoyé via TCP vers {target_device}")
        elif protocol == "udp":
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(data, (target_device, 12345))
            s.close()
            print(f"[Transmission] Bloc compressé envoyé via UDP vers {target_device}")
        elif protocol == "sfp":
            print(f"[Transmission] (SFP+) : stub, à implémenter selon firmware.")
        elif protocol == "rdma":
            print(f"[Transmission] (RDMA) : stub, à intégrer avec librairie RDMA.")
        elif protocol == "usb4":
            print(f"[Transmission] (USB4) : stub, à intégrer avec driver USB4 AMD AI HX.")
        else:
            print(f"[Transmission] Protocole custom non reconnu : {protocol}")

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
