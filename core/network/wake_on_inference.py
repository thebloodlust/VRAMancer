"""
VRAMancer V2: Wake-on-Inference (Le Cluster Écologique)
-------------------------------------------------------
Réveille les nœuds endormis (Mac, PC Portable) uniquement lorsqu'une
requête d'inférence nécessite leur VRAM. Utilise le protocole Wake-on-LAN (WoL).
"""

import socket
import struct
import time
from typing import Dict

class WakeOnInference:
    def __init__(self):
        # Registre des adresses MAC des nœuds du cluster
        self.node_macs: Dict[str, str] = {
            "laptop-i5": "00:1A:2B:3C:4D:5E",
            "mac-mini-m4": "A1:B2:C3:D4:E5:F6"
        }
        self.node_status: Dict[str, str] = {
            "laptop-i5": "sleep",
            "mac-mini-m4": "sleep"
        }

    def _create_magic_packet(self, macaddress: str) -> bytes:
        """Crée un paquet magique WoL à partir d'une adresse MAC."""
        if len(macaddress) == 17:
            macaddress = macaddress.replace(macaddress[2], '')
        if len(macaddress) != 12:
            raise ValueError("Adresse MAC invalide")
            
        data = b'FFFFFFFFFFFF' + (macaddress * 16).encode()
        send_data = b''
        for i in range(0, len(data), 2):
            send_data += struct.pack('B', int(data[i: i + 2], 16))
        return send_data

    def wake_node(self, node_id: str, broadcast_ip: str = '255.255.255.255', port: int = 9):
        """Envoie le paquet magique pour réveiller un nœud spécifique."""
        if node_id not in self.node_macs:
            print(f"[WoL] Nœud inconnu : {node_id}")
            return False
            
        mac = self.node_macs[node_id]
        packet = self.create_magic_packet(mac)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(packet, (broadcast_ip, port))
            sock.close()
            
            print(f"[WoL] ⚡ Magic Packet envoyé à {node_id} ({mac}). Réveil en cours...")
            self.node_status[node_id] = "waking_up"
            return True
        except Exception as e:
            print(f"[WoL] Erreur lors du réveil de {node_id}: {e}")
            return False

    def ensure_capacity(self, required_vram_mb: int):
        """Vérifie si la VRAM actuelle est suffisante, sinon réveille des nœuds."""
        # Logique simplifiée pour le PoC
        print(f"[WoL] Requête nécessitant {required_vram_mb} MB de VRAM reçue.")
        print("[WoL] Capacité locale insuffisante. Réveil du cluster étendu...")
        
        for node_id, status in self.node_status.items():
            if status == "sleep":
                self.wake_node(node_id)
                
        # Attendre que les nœuds se connectent via ClusterDiscovery
        # time.sleep(5) 

# Instance globale
wol_manager = WakeOnInference()
