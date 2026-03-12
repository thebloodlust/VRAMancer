"""Wake-on-Inference (WoI) - Module de réveil dynamique des nœuds GPU.

Permet de réveiller un Worker distant (via Wake-On-LAN) juste avant qu'une inférence 
lui soit envoyée, afin d'économiser l'énergie en gardant les GPUs en veille S3 
quand ils ne sont pas sollicités.
"""

import socket
import binascii
from core.logger import get_logger

log = get_logger("woi_engine")

def send_magic_packet(mac_address: str, ip_address: str = "255.255.255.255", port: int = 9) -> bool:
    """Envoie un Magic Packet (WOL) pour réveiller un Nœud distant.
    
    Args:
        mac_address (str): Adresse MAC de la carte réseau (ex: '00:1A:2B:3C:4D:5E')
        ip_address (str): Adresse IP de broadcast de la cible.
        port (int): Port UDP (9 par défaut pour WOL).
        
    Returns:
        bool: True si envoye avec succès, False sinon.
    """
    try:
        # Nettoyage de l'adresse MAC
        mac_clean = mac_address.replace(':', '').replace('-', '')
        if len(mac_clean) != 12:
            log.error(f"Adresse MAC invalide pour WoI : {mac_address}")
            return False

        # Construction du Magic Packet : 6 octets à FF suivis de 16 fois la MAC address
        data = b'FF' * 6 + (mac_clean * 16).encode('utf-8')
        magic_packet = binascii.unhexlify(data)

        # Envoi via socket UDP Broadcast
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(magic_packet, (ip_address, port))
        sock.close()
        
        log.info(f"✨ Magic Packet envoyé (Wake-on-Inference) vers MAC: {mac_address} sur IP: {ip_address}")
        return True
        
    except Exception as e:
        log.error(f"Echec du Wake-on-Inference : {e}")
        return False

# ----- TEST / UTILITAIRE -----
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mac = sys.argv[1]
        print(f"Tentative de réveil de {mac}...")
        send_magic_packet(mac)
    else:
        print("Usage: python wake_on_lan.py <MA:CA:DD:RE:SS>")