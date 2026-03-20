"""AITP NAT Traversal — Makes IPv6 anycast work for home networks.

Problem:  Real BGP anycast requires ASN + BGP peering — impossible for home users.
          IPv6 may be behind NAT66/NAT64 (CGNAT), or absent entirely.

Solution: Three-layer traversal strategy (auto-selected):

  1. Direct IPv6:  If both peers have native routable IPv6, use it.
                   Detected via STUN6 or by probing well-known IPv6 targets.

  2. UDP Hole Punch:  If one/both peers are behind stateful NAT, use STUN
                      to discover reflexive addresses and perform simultaneous
                      UDP open.  Works with full-cone and restricted NAT.

  3. Relay:  If UDP hole punch fails (symmetric NAT), traffic is relayed
             through a known relay node.  Any VRAMancer master node can act
             as a relay.

  4. IPv6 over IPv4 Tunnel:  If no IPv6 at all, auto-configure a 6in4 or
             Teredo-style encapsulation for LAN-only operation.

Usage:
    traversal = NATTraversal()
    my_addr = traversal.discover_external()
    traversal.punch_hole(peer_addr, peer_port)
    # or
    traversal.relay_via(relay_addr, peer_uid, tensor_bytes)
"""

import os
import sys
import socket
import struct
import logging
import time
import threading
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Well-known STUN servers (IPv6-capable)
STUN_SERVERS = [
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
    ("stun.stunprotocol.org", 3478),
]

# STUN message constants (RFC 5389)
STUN_BINDING_REQUEST = 0x0001
STUN_BINDING_RESPONSE = 0x0101
STUN_MAGIC_COOKIE = 0x2112A442
STUN_ATTR_MAPPED_ADDRESS = 0x0001
STUN_ATTR_XOR_MAPPED_ADDRESS = 0x0020

AITP_PORT = 9109
AITP_RELAY_PORT = 9111


class NATTraversal:
    """Discovers external address and performs hole punching for AITP peers."""

    def __init__(self, local_port: int = AITP_PORT):
        self.local_port = local_port
        self._external_ipv6: Optional[str] = None
        self._external_ipv4: Optional[str] = None
        self._nat_type: Optional[str] = None  # "none", "full_cone", "restricted", "symmetric"
        self._relay_addr: Optional[str] = None
        self._punched_peers: Dict[str, Tuple[str, int]] = {}  # uid -> (addr, port)

    # ------------------------------------------------------------------
    # IPv6 availability detection
    # ------------------------------------------------------------------

    @staticmethod
    def has_ipv6() -> bool:
        """Check if the host has a routable IPv6 address."""
        if not socket.has_ipv6:
            return False
        try:
            # Try connecting to a well-known IPv6 address (Google DNS)
            s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            s.settimeout(2.0)
            s.connect(("2001:4860:4860::8888", 53))
            addr = s.getsockname()[0]
            s.close()
            # Link-local (fe80::) doesn't count
            return not addr.startswith("fe80") and not addr.startswith("::1")
        except Exception:
            return False

    @staticmethod
    def get_local_ipv6() -> Optional[str]:
        """Get the best local IPv6 address (non-link-local)."""
        try:
            s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            s.settimeout(2.0)
            s.connect(("2001:4860:4860::8888", 53))
            addr = s.getsockname()[0]
            s.close()
            if not addr.startswith("fe80") and not addr.startswith("::1"):
                return addr
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # STUN client (minimal RFC 5389)
    # ------------------------------------------------------------------

    def _stun_request(self, family: int = socket.AF_INET) -> Optional[Tuple[str, int]]:
        """Send STUN Binding Request, return (external_ip, external_port)."""
        import secrets
        transaction_id = secrets.token_bytes(12)

        # Build STUN Binding Request
        msg = struct.pack(
            "!HHI12s",
            STUN_BINDING_REQUEST,
            0,  # length (no attributes)
            STUN_MAGIC_COOKIE,
            transaction_id,
        )

        for server, port in STUN_SERVERS:
            try:
                infos = socket.getaddrinfo(server, port, family, socket.SOCK_DGRAM)
                if not infos:
                    continue
                af, stype, proto, _, sa = infos[0]
                sock = socket.socket(af, stype, proto)
                sock.settimeout(3.0)
                sock.sendto(msg, sa)
                data, _ = sock.recvfrom(1024)
                sock.close()

                if len(data) < 20:
                    continue

                msg_type, msg_len, magic = struct.unpack("!HHI", data[:8])
                if msg_type != STUN_BINDING_RESPONSE:
                    continue

                # Parse attributes
                offset = 20
                while offset + 4 <= len(data):
                    attr_type, attr_len = struct.unpack("!HH", data[offset : offset + 4])
                    attr_data = data[offset + 4 : offset + 4 + attr_len]

                    if attr_type == STUN_ATTR_XOR_MAPPED_ADDRESS and len(attr_data) >= 8:
                        attr_family = attr_data[1]
                        xport = struct.unpack("!H", attr_data[2:4])[0] ^ (STUN_MAGIC_COOKIE >> 16)
                        if attr_family == 0x01:  # IPv4
                            xip_int = struct.unpack("!I", attr_data[4:8])[0] ^ STUN_MAGIC_COOKIE
                            xip = socket.inet_ntoa(struct.pack("!I", xip_int))
                            return (xip, xport)
                        elif attr_family == 0x02 and len(attr_data) >= 20:  # IPv6
                            xor_bytes = struct.pack("!I12s", STUN_MAGIC_COOKIE, transaction_id)
                            ip_bytes = bytes(a ^ b for a, b in zip(attr_data[4:20], xor_bytes))
                            xip = socket.inet_ntop(socket.AF_INET6, ip_bytes)
                            return (xip, xport)

                    if attr_type == STUN_ATTR_MAPPED_ADDRESS and len(attr_data) >= 8:
                        attr_family = attr_data[1]
                        mport = struct.unpack("!H", attr_data[2:4])[0]
                        if attr_family == 0x01:
                            mip = socket.inet_ntoa(attr_data[4:8])
                            return (mip, mport)
                        elif attr_family == 0x02 and len(attr_data) >= 20:
                            mip = socket.inet_ntop(socket.AF_INET6, attr_data[4:20])
                            return (mip, mport)

                    # Pad to 4-byte boundary
                    offset += 4 + ((attr_len + 3) // 4) * 4
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # External address discovery
    # ------------------------------------------------------------------

    def discover_external(self) -> Dict[str, Optional[str]]:
        """Discover external IPv4 and IPv6 addresses via STUN.

        Returns dict with keys: ``ipv6``, ``ipv4``, ``nat_type``.
        """
        result = {"ipv6": None, "ipv4": None, "nat_type": "unknown"}

        # Try native IPv6 first
        local_v6 = self.get_local_ipv6()
        if local_v6:
            stun_v6 = self._stun_request(socket.AF_INET6)
            if stun_v6:
                result["ipv6"] = stun_v6[0]
                if stun_v6[0] == local_v6:
                    result["nat_type"] = "none"
                else:
                    result["nat_type"] = "nat66"
            else:
                result["ipv6"] = local_v6
                result["nat_type"] = "none"

        # Always try IPv4 STUN (for tunnel fallback)
        stun_v4 = self._stun_request(socket.AF_INET)
        if stun_v4:
            result["ipv4"] = stun_v4[0]

        self._external_ipv6 = result["ipv6"]
        self._external_ipv4 = result["ipv4"]
        self._nat_type = result["nat_type"]

        logger.info(
            f"[NAT] External: IPv6={result['ipv6']}, IPv4={result['ipv4']}, "
            f"NAT={result['nat_type']}"
        )
        return result

    # ------------------------------------------------------------------
    # UDP hole punching
    # ------------------------------------------------------------------

    def punch_hole(
        self,
        peer_addr: str,
        peer_port: int = AITP_PORT,
        attempts: int = 5,
        interval: float = 0.5,
    ) -> bool:
        """Perform UDP hole punch to a known peer address.

        Both sides should call this simultaneously (coordinated via sensing
        or relay signaling).  Sends empty AITP probes to open the NAT
        pinhole.

        Returns True if we received a probe back (hole punched).
        """
        try:
            family = socket.AF_INET6 if ":" in peer_addr else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(interval)
            bind_addr = "::" if family == socket.AF_INET6 else "0.0.0.0"
            sock.bind((bind_addr, self.local_port))
        except OSError as e:
            logger.warning(f"[NAT] Hole punch bind failed: {e}")
            return False

        # Probe packet: AITP header with empty payload + HMAC
        import hmac as _hmac, hashlib as _hl
        from core.network.aitp_protocol import AITP_HEADER_FORMAT, AITP_MAGIC
        probe_header = struct.pack(AITP_HEADER_FORMAT, AITP_MAGIC, 1, 0x01, 0, 0)
        secret = os.environ.get("VRM_CLUSTER_SECRET", os.environ.get("VRM_API_TOKEN", "default_secret")).encode()
        sig = _hmac.new(secret, probe_header, _hl.sha256).digest()
        probe = probe_header + sig

        success = False
        for i in range(attempts):
            try:
                sock.sendto(probe, (peer_addr, peer_port))
                data, addr = sock.recvfrom(256)
                if data[:2] == AITP_MAGIC:
                    logger.info(f"[NAT] Hole punch success with {peer_addr} (attempt {i + 1})")
                    self._punched_peers[peer_addr] = (addr[0], addr[1])
                    success = True
                    break
            except socket.timeout:
                continue
            except Exception:
                break
        sock.close()
        return success

    # ------------------------------------------------------------------
    # Relay mode (for symmetric NAT)
    # ------------------------------------------------------------------

    def set_relay(self, relay_addr: str):
        """Set a relay node for traffic that can't be hole-punched."""
        self._relay_addr = relay_addr
        logger.info(f"[NAT] Relay node set: {relay_addr}")

    def send_via_relay(self, peer_uid: str, data: bytes):
        """Send data to peer via the relay node.

        Relay protocol: 4-byte peer_uid_len + peer_uid + data
        """
        if not self._relay_addr:
            logger.warning("[NAT] No relay configured")
            return False
        try:
            family = socket.AF_INET6 if ":" in self._relay_addr else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_DGRAM)
            uid_bytes = peer_uid.encode("utf-8")
            relay_packet = struct.pack("!I", len(uid_bytes)) + uid_bytes + data
            sock.sendto(relay_packet, (self._relay_addr, AITP_RELAY_PORT))
            sock.close()
            return True
        except Exception as e:
            logger.warning(f"[NAT] Relay send failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Best-effort send: direct > punched > relay
    # ------------------------------------------------------------------

    def send_to_peer(self, peer_uid: str, peer_addr: str, data: bytes) -> str:
        """Send data to peer using the best available path.

        Returns the method used: ``"direct"``, ``"punched"``, ``"relay"``, or ``"failed"``.
        """
        # 1. Try direct
        try:
            family = socket.AF_INET6 if ":" in peer_addr else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_DGRAM)
            sock.sendto(data, (peer_addr, self.local_port))
            sock.close()
            return "direct"
        except Exception:
            pass

        # 2. Try punched hole
        if peer_addr in self._punched_peers:
            try:
                addr, port = self._punched_peers[peer_addr]
                family = socket.AF_INET6 if ":" in addr else socket.AF_INET
                sock = socket.socket(family, socket.SOCK_DGRAM)
                sock.sendto(data, (addr, port))
                sock.close()
                return "punched"
            except Exception:
                pass

        # 3. Relay
        if self._relay_addr and self.send_via_relay(peer_uid, data):
            return "relay"

        return "failed"

    # ------------------------------------------------------------------
    # LAN-only IPv6 tunnel (when no native IPv6)
    # ------------------------------------------------------------------

    @staticmethod
    def create_lan_ipv6_overlay(subnet_prefix: str = "fd42:vrm:swarm") -> Optional[str]:
        """Generate a ULA (Unique Local Address) for LAN-only IPv6 swarm.

        Uses fd00::/8 private IPv6 space like 192.168.x.x for IPv4.
        No tunnel or internet IPv6 needed — pure link-local overlay.

        Returns the generated address string, e.g. ``fd42:vrm:swarm::1``.
        """
        import uuid
        # Generate interface ID from machine UUID
        node_id = uuid.getnode()
        iid = f"{node_id:012x}"
        addr = f"{subnet_prefix}::{iid[:4]}:{iid[4:8]}:{iid[8:]}"
        logger.info(f"[NAT] LAN IPv6 overlay address: {addr}")
        return addr

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        return {
            "external_ipv6": self._external_ipv6,
            "external_ipv4": self._external_ipv4,
            "nat_type": self._nat_type,
            "relay": self._relay_addr,
            "punched_peers": list(self._punched_peers.keys()),
            "has_native_ipv6": self.has_ipv6(),
        }
