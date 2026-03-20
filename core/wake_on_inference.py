import socket
import logging
import time

logger = logging.getLogger("vramancer.wake")

class WakeOnInferenceManager:
    """Wake ON Inference (WOI) Manager."""
    def __init__(self):
        self.known_macs = set()
        
    def register_node(self, mac_address: str):
        if not mac_address:
            return
        mac_clean = mac_address.replace(':', '').replace('-', '')
        if len(mac_clean) == 12:
            self.known_macs.add(mac_clean)
            
    def wake_all(self):
        if not self.known_macs:
            return
            
        logger.info(f" [Wake On Inference] Waking up {len(self.known_macs)} sleeping nodes via WoL...")
        for mac in self.known_macs:
            self._send_magic_packet(mac)
            
    def _send_magic_packet(self, mac_address: str):
        try:
            data = bytes.fromhex('F' * 12 + mac_address * 16)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(data, ('<broadcast>', 9))
            sock.close()
        except Exception as e:
            pass

_manager = WakeOnInferenceManager()

def get_woi_manager() -> WakeOnInferenceManager:
    return _manager
