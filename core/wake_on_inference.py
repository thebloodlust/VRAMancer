import socket
import logging
import time

logger = logging.getLogger("vramancer.wake")

class WakeOnInferenceManager:
    """Wake ON Inference (WOI) Manager.

    Wakes sleeping Swarm nodes via Wake-on-LAN magic packets before
    inference begins, so that the full cluster is available.

    Usage::

        woi = get_woi_manager()
        woi.register_node("AA:BB:CC:DD:EE:FF")
        woi.wake_all()              # broadcast WoL to all registered nodes
        woi.wait_for_nodes(timeout=30)   # optional: wait until nodes respond
    """

    def __init__(self):
        self.known_macs = set()
        self._wake_history: list = []  # timestamps of wake_all() calls

    def register_node(self, mac_address: str):
        if not mac_address:
            return
        mac_clean = mac_address.replace(':', '').replace('-', '').lower()
        if len(mac_clean) == 12:
            self.known_macs.add(mac_clean)

    def unregister_node(self, mac_address: str):
        mac_clean = mac_address.replace(':', '').replace('-', '').lower()
        self.known_macs.discard(mac_clean)

    def wake_all(self, subnet: str = '<broadcast>', port: int = 9):
        """Send WoL magic packets to all registered nodes.

        Parameters
        ----------
        subnet : str
            Broadcast address. Defaults to '<broadcast>' (local LAN).
            Set to e.g. '192.168.1.255' for a specific subnet.
        port : int
            UDP port (standard WoL uses 7 or 9).
        """
        if not self.known_macs:
            return 0

        logger.info(
            "[Wake On Inference] Waking %d sleeping nodes via WoL (target=%s:%d)",
            len(self.known_macs), subnet, port,
        )
        count = 0
        for mac in self.known_macs:
            if self._send_magic_packet(mac, subnet, port):
                count += 1
        self._wake_history.append({"ts": time.time(), "nodes": count})
        return count

    def _send_magic_packet(self, mac_address: str, subnet: str = '<broadcast>', port: int = 9) -> bool:
        try:
            data = bytes.fromhex('FF' * 6 + mac_address * 16)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(data, (subnet, port))
            sock.close()
            return True
        except Exception as e:
            logger.debug("WoL packet failed for %s: %s", mac_address, e)
            return False

    def wait_for_nodes(self, expected: int = 0, timeout: float = 30.0,
                       discovery=None) -> int:
        """Wait for cluster nodes to come online after wake.

        Parameters
        ----------
        expected : int
            Expected number of nodes (0 = len(known_macs)).
        timeout : float
            Maximum wait time in seconds.
        discovery : ClusterDiscovery, optional
            If provided, poll its node count.

        Returns
        -------
        int
            Number of nodes online when we stopped waiting.
        """
        if expected <= 0:
            expected = len(self.known_macs)
        if discovery is None:
            # Without discovery, just sleep a reasonable boot time
            time.sleep(min(timeout, 10))
            return 0

        deadline = time.time() + timeout
        while time.time() < deadline:
            n = discovery.node_count()
            if n >= expected:
                logger.info("[WOI] %d/%d nodes online", n, expected)
                return n
            time.sleep(2)
        n = discovery.node_count()
        logger.warning("[WOI] Timeout: %d/%d nodes online after %.0fs", n, expected, timeout)
        return n

    @property
    def stats(self) -> dict:
        return {
            "registered_macs": len(self.known_macs),
            "wake_history": self._wake_history[-10:],
        }

_manager = WakeOnInferenceManager()

def get_woi_manager() -> WakeOnInferenceManager:
    return _manager
