"""AITP Sensing — IPv6 Multicast node discovery and active heartbeat.

Each node announces its UID, hardware specs, and NAT info to a site-local
multicast group.  Peers are tracked with staleness eviction.  An active
heartbeat thread periodically broadcasts presence and probes for dead peers.
"""

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

AITP_SENSING_GROUP = "ff02::1:ff00:1"  # link-local multicast (VRAMancer)
AITP_SENSING_PORT = 9110

# Heartbeat interval (seconds)
_HEARTBEAT_INTERVAL = int(os.environ.get("VRM_SENSING_HEARTBEAT", "10"))
_PEER_MAX_AGE = int(os.environ.get("VRM_SENSING_PEER_TTL", "30"))

# ── Prometheus metrics (lazy) ──────────────────────────────────────────
_SENSING_PEERS = None
_SENSING_JOINS = None
_SENSING_LEAVES = None
_SENSING_PACKETS = None

def _init_sensing_metrics():
    global _SENSING_PEERS, _SENSING_JOINS, _SENSING_LEAVES, _SENSING_PACKETS
    if _SENSING_PEERS is not None:
        return
    try:
        from prometheus_client import Gauge, Counter
        _SENSING_PEERS = Gauge(
            "vramancer_sensing_active_peers",
            "Number of active AITP sensing peers",
        )
        _SENSING_JOINS = Counter(
            "vramancer_sensing_joins_total",
            "Peer join events",
        )
        _SENSING_LEAVES = Counter(
            "vramancer_sensing_leaves_total",
            "Peer eviction events (stale)",
        )
        _SENSING_PACKETS = Counter(
            "vramancer_sensing_packets_total",
            "Sensing packets processed",
            ["direction"],  # send, recv
        )
    except Exception:
        pass


def _get_cluster_secret() -> bytes:
    return os.environ.get(
        "VRM_CLUSTER_SECRET",
        os.environ.get("VRM_API_TOKEN", "default_secret"),
    ).encode("utf-8")


class AITPSensor:
    """IPv6 Multicast node discovery with active heartbeat."""

    def __init__(self, node_uid: str = None, hw_specs: dict = None):
        self.node_uid = node_uid or str(uuid.uuid4())
        self.hw_specs = hw_specs or {"type": "cpu", "compute": 0, "vram": 0}
        self.peers = {}  # uid -> {ipv6, hw, last_seen, nat}
        self.running = False
        self._nat = None
        self._external_info = {}
        self._heartbeat_thread = None
        self.sock = None

        _init_sensing_metrics()

    def _init_nat_traversal(self):
        try:
            from core.network.nat_traversal import NATTraversal
            self._nat = NATTraversal()
            self._external_info = self._nat.discover_external()
            if not self._external_info.get("ipv6") and not self._nat.has_ipv6():
                overlay = self._nat.create_lan_ipv6_overlay()
                self._external_info["ipv6_overlay"] = overlay
                logger.info(f"[Sensing] Using ULA overlay: {overlay}")
        except Exception as e:
            logger.debug(f"[Sensing] NAT traversal skipped: {e}")

    def get_discovery_payload(self) -> bytes:
        payload_data = {
            "uid": self.node_uid,
            "hw": self.hw_specs,
            "ts": time.time(),
            "nat": self._external_info,
        }
        json_data = json.dumps(payload_data).encode('utf-8')
        sig = hmac.new(
            _get_cluster_secret(), json_data, hashlib.sha256,
        ).hexdigest()
        signed_payload = {"sig": sig, "data": payload_data}
        return json.dumps(signed_payload).encode('utf-8')

    def start_listening(self):
        self.running = True
        self._init_nat_traversal()
        self.sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.sock.bind(('', AITP_SENSING_PORT))
            group_bin = socket.inet_pton(socket.AF_INET6, AITP_SENSING_GROUP)
            mreq = group_bin + struct.pack('@I', 0)
            self.sock.setsockopt(
                socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq,
            )
            logger.info(
                f"AITP Sensing active on {AITP_SENSING_GROUP}:{AITP_SENSING_PORT}"
            )

            self.listener_thread = threading.Thread(
                target=self._listen_loop, daemon=True, name="AITP-Sensing-RX",
            )
            self.listener_thread.start()

            # Start active heartbeat
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True, name="AITP-Sensing-HB",
            )
            self._heartbeat_thread.start()

        except Exception as e:
            logger.warning(f"AITP Sensing bind failed: {e}")

    def _listen_loop(self):
        secret = _get_cluster_secret()
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                signed_info = json.loads(data.decode('utf-8'))

                if "sig" not in signed_info or "data" not in signed_info:
                    continue

                payload_data = signed_info["data"]
                sig_received = signed_info["sig"]
                json_data = json.dumps(payload_data).encode('utf-8')

                sig_calc = hmac.new(
                    secret, json_data, hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(sig_received, sig_calc):
                    logger.warning(f"[Sensing] Invalid HMAC from {addr[0]}")
                    continue

                uid = payload_data.get("uid")
                if uid and uid != self.node_uid:
                    is_new = uid not in self.peers
                    self.peers[uid] = {
                        "ipv6": addr[0],
                        "hw": payload_data.get("hw", {}),
                        "last_seen": time.time(),
                        "nat": payload_data.get("nat", {}),
                    }
                    if is_new:
                        logger.info(f"[Sensing] New peer: {uid} ({payload_data.get('hw')})")
                        if _SENSING_JOINS:
                            _SENSING_JOINS.inc()
                    if _SENSING_PACKETS:
                        _SENSING_PACKETS.labels("recv").inc()

            except Exception as e:
                logger.debug(f"[Sensing] packet drop: {e}")

    def _heartbeat_loop(self):
        """Periodically broadcast presence and evict stale peers."""
        while self.running:
            try:
                self.broadcast_presence()
                self._evict_stale_peers()
            except Exception as e:
                logger.debug(f"[Sensing] heartbeat error: {e}")
            time.sleep(_HEARTBEAT_INTERVAL)

    def _evict_stale_peers(self):
        """Remove peers that haven't been seen within the TTL."""
        now = time.time()
        stale = [
            uid for uid, info in self.peers.items()
            if now - info["last_seen"] > _PEER_MAX_AGE
        ]
        for uid in stale:
            del self.peers[uid]
            logger.info(f"[Sensing] Evicted stale peer: {uid}")
            if _SENSING_LEAVES:
                _SENSING_LEAVES.inc()
        if _SENSING_PEERS:
            _SENSING_PEERS.set(len(self.peers))

    def broadcast_presence(self):
        if not self.sock:
            return
        try:
            payload = self.get_discovery_payload()
            self.sock.sendto(payload, (AITP_SENSING_GROUP, AITP_SENSING_PORT))
            if _SENSING_PACKETS:
                _SENSING_PACKETS.labels("send").inc()
        except Exception as e:
            logger.debug(f"[Sensing] broadcast failed: {e}")

    def get_available_peers(self, max_age=None):
        now = time.time()
        age = max_age if max_age is not None else _PEER_MAX_AGE
        return {
            k: v for k, v in self.peers.items()
            if now - v["last_seen"] < age
        }

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        logger.info("[Sensing] Stopped")
