import socket
import struct
import json
import time
import uuid
import threading
import logging
import os
import hmac
import hashlib

logger = logging.getLogger(__name__)

# IPv6 Multicast Group for VRAMancer Node Discovery (Sensing)
AITP_SENSING_GROUP = "ff12::a1:b2:c3"  # Site-local multicast 
AITP_SENSING_PORT = 9110

def _get_cluster_secret() -> bytes:
    return os.environ.get("VRM_CLUSTER_SECRET", os.environ.get("VRM_API_TOKEN", "default_secret")).encode("utf-8")

class AITPSensor:
    """
    Protocole de découverte (Sensing) AITP via IPv6 Multicast.
    Équivalent du protocole ARP/NDP, mais pour le hardware IA.
    Permet à un noeud d'annoncer ses TFLOPS, sa VRAM, et son UID 
    sans l'intervention d'un serveur central.
    Sécurisé par signature HMAC.

    Auto-discovers NAT type and external address at startup.
    Falls back to ULA (fd42:vrm:swarm::) overlay when no native IPv6.
    """
    def __init__(self, node_uid: str = None, hw_specs: dict = None):
        self.node_uid = node_uid or str(uuid.uuid4())
        self.hw_specs = hw_specs or {"type": "cpu", "compute": 0, "vram": 0}
        self.peers = {} # uid -> {specs, last_seen}
        self.running = False
        self._nat: object = None
        self._external_info: dict = {}

    def _init_nat_traversal(self):
        """Lazy-init NAT traversal and discover external address."""
        try:
            from core.network.nat_traversal import NATTraversal
            self._nat = NATTraversal()
            self._external_info = self._nat.discover_external()
            # If no IPv6 at all, set up LAN overlay
            if not self._external_info.get("ipv6") and not self._nat.has_ipv6():
                overlay = self._nat.create_lan_ipv6_overlay()
                self._external_info["ipv6_overlay"] = overlay
                logger.info(f"[Sensing] No native IPv6, using ULA overlay: {overlay}")
        except Exception as e:
            logger.debug(f"[Sensing] NAT traversal init skipped: {e}")

    def get_discovery_payload(self) -> bytes:
        payload_data = {
            "uid": self.node_uid,
            "hw": self.hw_specs,
            "ts": time.time(),
            "nat": self._external_info,  # includes external ipv6/ipv4/nat_type
        }
        json_data = json.dumps(payload_data).encode('utf-8')
        sig = hmac.new(_get_cluster_secret(), json_data, hashlib.sha256).hexdigest()
        signed_payload = {"sig": sig, "data": payload_data}
        return json.dumps(signed_payload).encode('utf-8')

    def start_listening(self):
        self.running = True
        self._init_nat_traversal()
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
        secret = _get_cluster_secret()
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                signed_info = json.loads(data.decode('utf-8'))
                
                # Check for old format or missing signature
                if "sig" not in signed_info or "data" not in signed_info:
                    continue
                
                payload_data = signed_info["data"]
                sig_received = signed_info["sig"]
                json_data = json.dumps(payload_data).encode('utf-8')
                
                # Verify HMAC
                sig_calc = hmac.new(secret, json_data, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(sig_received, sig_calc):
                    logger.warning(f"Signature AITP Sensing invalide de {addr[0]}")
                    continue

                uid = payload_data.get("uid")
                if uid and uid != self.node_uid:
                    # Nouveau noeud découvert ou mis à jour !
                    self.peers[uid] = {
                        "ipv6": addr[0],
                        "hw": payload_data.get("hw", {}),
                        "last_seen": time.time(),
                        "nat": payload_data.get("nat", {}),
                    }
                    logger.debug(f"[AITP Sensing] Noeud détecté: {uid} ({payload_data.get('hw')})")
            except Exception as e:
                logger.debug(f"AITP Sensing packet drop: {e}")

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
