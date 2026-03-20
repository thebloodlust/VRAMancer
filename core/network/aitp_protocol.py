import socket
import struct
import logging
import uuid
import json
import os
import hmac
import hashlib

logger = logging.getLogger(__name__)

# AITP Header Format (16 bytes):
# [ Magic (2 bytes) | Version (1 byte) | Flags (1 byte) | Layer ID (4 bytes) | Tensor Size (8 bytes) ]
AITP_HEADER_FORMAT = "!2sBBQI"
AITP_MAGIC = b"VT"
AITP_VERSION = 1

def _get_cluster_secret() -> bytes:
    return os.environ.get("VRM_CLUSTER_SECRET", os.environ.get("VRM_API_TOKEN", "default_secret")).encode("utf-8")

class AITPProtocol:
    """
    VRAMancer Tensor Protocol (AITP)
    Implémentation de base pour l'encapsulation UDP et le routage Anycast IPv6.
    Conçu pour contourner le Kernel OS (via eBPF/XDP à terme) et écrire directement en VRAM.
    """
    
    def __init__(self, port=9109, anycast_ipv6="ff02::vrm"):
        self.port = port
        self.anycast_ipv6 = anycast_ipv6
        self.sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.sock.bind(('::', self.port))
            logger.info(f"AITP Protocol initialisé sur le port IPv6 {self.port} (UDP Anycast Ready)")
        except Exception as e:
            logger.error(f"Erreur de bind IPv6 pour AITP: {e}")

    def create_packet(self, layer_id: int, tensor_bytes: bytes, flags: int = 0) -> bytes:
        """Crée un datagramme AITP optimisé"""
        size = len(tensor_bytes)
        # Header binaire ultra-rapide
        header = struct.pack(
            AITP_HEADER_FORMAT,
            AITP_MAGIC,
            AITP_VERSION,
            flags,
            size,
            layer_id
        )
        packet_body = header + tensor_bytes
        # Signature
        sig = hmac.new(_get_cluster_secret(), packet_body, hashlib.sha256).digest()
        # Tout en un seul paquet (idéal avec un MTU Jumbo Frames 9000 bytes) + 32 bytes signature
        return packet_body + sig

    def parse_packet(self, packet: bytes):
        """Parse un paquet AITP entrant à la vitesse de l'éclair"""
        header_size = struct.calcsize(AITP_HEADER_FORMAT)
        if len(packet) < header_size + 32:
            raise ValueError("AITP Packet trop petit (signature manquante)")
            
        # Verify HMCA
        packet_body = packet[:-32]
        received_sig = packet[-32:]
        expected_sig = hmac.new(_get_cluster_secret(), packet_body, hashlib.sha256).digest()
        if not hmac.compare_digest(received_sig, expected_sig):
            raise ValueError("Signature AITP invalide")

        header = packet_body[:header_size]
        magic, version, flags, size, layer_id = struct.unpack(AITP_HEADER_FORMAT, header)
        
        if magic != AITP_MAGIC:
            raise ValueError("Magic Number AITP invalide")
            
        tensor_data = packet_body[header_size:header_size+size]
        return {
            "version": version,
            "layer_id": layer_id,
            "flags": flags,
            "tensor_data": tensor_data
        }

    def send_anycast(self, routing_address: str, layer_id: int, tensor_bytes: bytes):
        """
        Envoie un tenseur via IPv6. 
        Si l'adresse est Anycast, le réseau l'achemine au GPU le plus proche/disponible.
        """
        packet = self.create_packet(layer_id, tensor_bytes)
        self.sock.sendto(packet, (routing_address, self.port))
        logger.debug(f"AITP Packet envoyé vers {routing_address} (Layer {layer_id})")

    def recv_loop(self, callback=None, timeout: float = 1.0):
        """Blocking receive loop — dispatches parsed tensors to callback.

        Args:
            callback: ``(layer_id, tensor_data, flags, addr) -> None``
            timeout:  socket timeout in seconds for graceful shutdown check.
        """
        self.sock.settimeout(timeout)
        self._recv_running = True
        logger.info(f"AITP recv_loop started on [::]:{self.port}")
        while self._recv_running:
            try:
                data, addr = self.sock.recvfrom(65535)
                parsed = self.parse_packet(data)
                if callback:
                    callback(parsed["layer_id"], parsed["tensor_data"], parsed["flags"], addr)
            except socket.timeout:
                continue
            except ValueError as e:
                logger.debug(f"AITP recv bad packet from {addr}: {e}")
            except Exception as e:
                logger.debug(f"AITP recv error: {e}")

    def stop_recv(self):
        """Signal the recv_loop to stop."""
        self._recv_running = False

# Singleton pour le noeud
_global_aitp = None

def get_aitp_protocol():
    global _global_aitp
    if _global_aitp is None:
        _global_aitp = AITPProtocol()
    return _global_aitp
