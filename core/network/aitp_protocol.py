import socket
import struct
import logging
import uuid
import json

logger = logging.getLogger(__name__)

# AITP Header Format (16 bytes):
# [ Magic (2 bytes) | Version (1 byte) | Flags (1 byte) | Layer ID (4 bytes) | Tensor Size (8 bytes) ]
AITP_HEADER_FORMAT = "!2sBBQI"
AITP_MAGIC = b"VT"
AITP_VERSION = 1

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
        # Tout en un seul paquet (idéal avec un MTU Jumbo Frames 9000 bytes)
        return header + tensor_bytes

    def parse_packet(self, packet: bytes):
        """Parse un paquet AITP entrant à la vitesse de l'éclair"""
        header_size = struct.calcsize(AITP_HEADER_FORMAT)
        if len(packet) < header_size:
            raise ValueError("AITP Packet trop petit")
            
        header = packet[:header_size]
        magic, version, flags, size, layer_id = struct.unpack(AITP_HEADER_FORMAT, header)
        
        if magic != AITP_MAGIC:
            raise ValueError("Magic Number AITP invalide")
            
        tensor_data = packet[header_size:header_size+size]
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

# Singleton pour le noeud
_global_aitp = None

def get_aitp_protocol():
    global _global_aitp
    if _global_aitp is None:
        _global_aitp = AITPProtocol()
    return _global_aitp
