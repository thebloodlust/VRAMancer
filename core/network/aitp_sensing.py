import socket
import struct
import json
import time
import uuid
import threading
import logging

logger = logging.getLogger(__name__)

# IPv6 Multicast Group for VRAMancer Node Discovery (Sensing)
AITP_SENSING_GROUP = "ff12::a1:b2:c3"  # Site-local multicast 
AITP_SENSING_PORT = 9110

class AITPSensor:
    """
    Protocole de découverte (Sensing) AITP via IPv6 Multicast.
    Équivalent du protocole ARP/NDP, mais pour le hardware IA.
    Permet à un noeud d'annoncer ses TFLOPS, sa VRAM, et son UID 
    sans l'intervention d'un serveur central.
    """
    def __init__(self, node_uid: str = None, hw_specs: dict = None):
        self.node_uid = node_uid or str(uuid.uuid4())
        self.hw_specs = hw_specs or {"type": "cpu", "compute": 0, "vram": 0}
        self.peers = {} # uid -> {specs, last_seen}
        self.running = False

    def get_discovery_payload(self) -> bytes:
        payload = {
            "uid": self.node_uid,
            "hw": self.hw_specs,
            "ts": time.time()
        }
        return json.dumps(payload).encode('utf-8')

    def start_listening(self):
        self.running = True
        self.sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.sock.bind(('', AITP_SENSING_PORT))
            # Join Multicast group
            group_bin = socket.inet_pton(socket.AF_INET6, AITP_SENSING_GROUP)
            # Interface index 0 (all)
            mreq = group_bin + struct.pack('@I', 0)
            self.sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)
            
            logger.info(f"AITP Sensing actif: Écoute sur {AITP_SENSING_GROUP}:{AITP_SENSING_PORT}")
            
            self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listener_thread.start()
        except Exception as e:
            logger.warning(f"Impossible de binder AITP Sensing (IPv6 local missing?): {e}")

    def _listen_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                info = json.loads(data.decode('utf-8'))
                uid = info.get("uid")
                if uid and uid != self.node_uid:
                    # Nouveau noeud découvert ou mis à jour !
                    self.peers[uid] = {
                        "ipv6": addr[0],
                        "hw": info.get("hw", {}),
                        "last_seen": time.time()
                    }
                    logger.debug(f"[AITP Sensing] Noeud détecté: {uid} ({info.get('hw')})")
            except Exception as e:
                pass

    def broadcast_presence(self):
        """Inonde le réseau local/ring pour dire: 'Je suis là et j'ai X VRAM'"""
        if not hasattr(self, 'sock'):
            return
        try:
            payload = self.get_discovery_payload()
            self.sock.sendto(payload, (AITP_SENSING_GROUP, AITP_SENSING_PORT))
        except Exception as e:
            logger.debug(f"Sensing broadcast failed: {e}")

    def get_available_peers(self, max_age=30):
        """Retourne la carte radar du réseau P2P local"""
        now = time.time()
        # Filtrer les noeuds trop vieux (déconnectés)
        active = {k: v for k, v in self.peers.items() if now - v["last_seen"] < max_age}
        return active
