"""AITP — VRAMancer Tensor Protocol.

Binary UDP protocol for shipping tensor shards between swarm nodes.
Supports optional FEC (Reed-Solomon) for lossy links.

Header (16 bytes):
  Magic (2B)  | Version (1B) | Flags (1B) | Tensor Size (8B) | Layer ID (4B)
  + payload + HMAC-SHA256 (32B)

Flag bits:
  0x01 — FEC-protected (payload is an FEC-encoded shard bundle)
  0x02 — Compressed (zstd)
  0x04 — Priority (route to fastest GPU)
"""

import socket
import struct
import logging
import os
import hmac
import hashlib
import time

logger = logging.getLogger(__name__)

AITP_HEADER_FORMAT = "!2sBBQI"
AITP_MAGIC = b"VT"
AITP_VERSION = 2
# v2 (2026-09-23) : la méta FEC porte un IDENTIFIANT DE MESSAGE. En v1, le réassemblage
# n'était indexé que par layer_id : deux tenseurs de la même couche en vol en même temps
# (cas normal du décodage pipeliné) mélangeaient leurs fragments et le récepteur livrait
# un tenseur CORROMPU sans erreur — mesuré par benchmarks/bench_aitp_protocol.py.
FEC_META_V1 = "!HHI"      # total_shards, shard_idx, original_size
FEC_META_V2 = "!HHII"     # + msg_id
# Un datagramme UDP ne dépasse pas 65 507 octets : au-delà, sendto() échoue.
MAX_DATAGRAM = 65507
FEC_REASSEMBLY_TIMEOUT_S = float(os.environ.get("VRM_AITP_FEC_TIMEOUT", "5"))

# ── Standard AITP port (configurable via VRM_AITP_PORT) ────────────────
AITP_PORT = int(os.environ.get("VRM_AITP_PORT", "55555"))

# Flag bits
FLAG_FEC = 0x01
FLAG_COMPRESSED = 0x02
FLAG_PRIORITY = 0x04

# ── Prometheus metrics (lazy) ──────────────────────────────────────────
_AITP_SENT = None
_AITP_RECV = None
_AITP_ERRORS = None
_AITP_BYTES = None
_AITP_LATENCY = None

def _init_aitp_metrics():
    global _AITP_SENT, _AITP_RECV, _AITP_ERRORS, _AITP_BYTES, _AITP_LATENCY
    if _AITP_SENT is not None:
        return
    try:
        from prometheus_client import Counter, Histogram
        _AITP_SENT = Counter(
            "vramancer_aitp_packets_sent_total",
            "AITP packets sent",
        )
        _AITP_RECV = Counter(
            "vramancer_aitp_packets_recv_total",
            "AITP packets received",
        )
        _AITP_ERRORS = Counter(
            "vramancer_aitp_errors_total",
            "AITP packet errors",
            ["kind"],  # hmac_fail, parse_error, timeout
        )
        _AITP_BYTES = Counter(
            "vramancer_aitp_bytes_total",
            "AITP payload bytes",
            ["direction"],  # send, recv
        )
        _AITP_LATENCY = Histogram(
            "vramancer_aitp_send_latency_seconds",
            "AITP send latency",
        )
    except Exception:
        logger.debug("AITP Prometheus metric registration failed", exc_info=True)


def _get_cluster_secret() -> bytes:
    return os.environ.get(
        "VRM_CLUSTER_SECRET",
        os.environ.get("VRM_API_TOKEN", "default_secret"),
    ).encode("utf-8")


class AITPProtocol:
    """UDP-based tensor transport with HMAC authentication and optional FEC."""

    def __init__(self, port=AITP_PORT, anycast_ipv6="ff02::1:ff00:1"):
        self.port = port
        self.anycast_ipv6 = anycast_ipv6
        self._fec = None
        self._recv_running = False
        self._msg_seq = 0
        self._seq_lock = __import__("threading").Lock()

        _init_aitp_metrics()

        self.sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.sock.bind(('::', self.port))
            logger.info(f"AITP bound on [::]:{self.port} (UDP)")
        except Exception as e:
            logger.error(f"AITP bind failed: {e}")

        # Join IPv6 multicast group for receiving anycast traffic
        try:
            mcast_addr = socket.inet_pton(socket.AF_INET6, self.anycast_ipv6)
            mreq = mcast_addr + struct.pack("@I", 0)  # interface 0 = all
            self.sock.setsockopt(
                socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq,
            )
            logger.info(f"AITP joined multicast group {self.anycast_ipv6}")
        except Exception as e:
            logger.debug(f"AITP multicast join skipped: {e}")

    # ── FEC integration ─────────────────────────────────────────────

    def enable_fec(self, data_shards: int = 10, parity_shards: int = 2):
        """Enable FEC protection on outgoing packets."""
        try:
            from experimental.aitp_fec import FastFEC
            self._fec = FastFEC(data_shards=data_shards, parity_shards=parity_shards)
            logger.info(f"AITP FEC enabled: {data_shards}+{parity_shards} RS GF(2^8)")
        except Exception as e:
            logger.warning(f"AITP FEC unavailable: {e}")

    # ── Packet creation / parsing ───────────────────────────────────

    def create_packet(self, layer_id: int, tensor_bytes: bytes, flags: int = 0) -> bytes:
        """Create a signed AITP packet."""
        size = len(tensor_bytes)
        header = struct.pack(
            AITP_HEADER_FORMAT,
            AITP_MAGIC,
            AITP_VERSION,
            flags,
            size,
            layer_id,
        )
        packet_body = header + tensor_bytes
        sig = hmac.new(_get_cluster_secret(), packet_body, hashlib.sha256).digest()
        return packet_body + sig

    def parse_packet(self, packet: bytes):
        """Parse and verify an incoming AITP packet."""
        header_size = struct.calcsize(AITP_HEADER_FORMAT)
        if len(packet) < header_size + 32:
            raise ValueError("AITP packet too small")

        packet_body = packet[:-32]
        received_sig = packet[-32:]
        expected_sig = hmac.new(_get_cluster_secret(), packet_body, hashlib.sha256).digest()
        if not hmac.compare_digest(received_sig, expected_sig):
            if _AITP_ERRORS:
                _AITP_ERRORS.labels("hmac_fail").inc()
            raise ValueError("AITP HMAC invalid")

        header = packet_body[:header_size]
        magic, version, flags, size, layer_id = struct.unpack(AITP_HEADER_FORMAT, header)

        if magic != AITP_MAGIC:
            raise ValueError("AITP magic invalid")

        tensor_data = packet_body[header_size:header_size + size]

        if _AITP_RECV:
            _AITP_RECV.inc()
        if _AITP_BYTES:
            _AITP_BYTES.labels("recv").inc(len(tensor_data))

        return {
            "version": version,
            "layer_id": layer_id,
            "flags": flags,
            "tensor_data": tensor_data,
        }

    # ── Send with optional FEC ──────────────────────────────────────

    def send_anycast(self, routing_address: str, layer_id: int, tensor_bytes: bytes):
        """Send a tensor shard via IPv6 UDP, optionally FEC-protected."""
        t0 = time.perf_counter()

        if self._fec is not None:
            # FEC encode: split into shards and send each as a separate packet
            shards = self._fec.encode(tensor_bytes)
            with self._seq_lock:
                msg_id = self._msg_seq
                self._msg_seq = (self._msg_seq + 1) & 0xFFFFFFFF
            for shard_idx, shard_data in enumerate(shards):
                meta = struct.pack(FEC_META_V2, len(shards), shard_idx,
                                   len(tensor_bytes), msg_id)
                packet = self.create_packet(layer_id, meta + shard_data, flags=FLAG_FEC)
                self._check_size(packet, len(tensor_bytes))
                self.sock.sendto(packet, (routing_address, self.port))
        else:
            packet = self.create_packet(layer_id, tensor_bytes)
            self._check_size(packet, len(tensor_bytes))
            self.sock.sendto(packet, (routing_address, self.port))

        if _AITP_SENT:
            _AITP_SENT.inc()
        if _AITP_BYTES:
            _AITP_BYTES.labels("send").inc(len(tensor_bytes))
        if _AITP_LATENCY:
            _AITP_LATENCY.observe(time.perf_counter() - t0)

        logger.debug(f"AITP sent to {routing_address} layer={layer_id} "
                     f"size={len(tensor_bytes)} fec={self._fec is not None}")

    def _check_size(self, packet: bytes, tensor_size: int) -> None:
        """Message clair plutôt qu'un `OSError: Message too long` venu du noyau."""
        if len(packet) > MAX_DATAGRAM:
            limit = "~64 Ko" if self._fec is None else f"~{64 * self._fec.data_shards} Ko"
            raise ValueError(
                f"AITP : tenseur de {tensor_size} octets trop gros pour UDP "
                f"(paquet de {len(packet)} > {MAX_DATAGRAM}). Limite actuelle {limit} "
                f"par tenseur ; AITP ne fragmente pas au-delà. Pour des activations de "
                f"prefill (plusieurs Mo), utiliser le transport TCP (VTP / RPC)."
            )

    # ── Receive loop ────────────────────────────────────────────────

    def recv_loop(self, callback=None, timeout: float = 1.0):
        """Blocking receive loop dispatching parsed tensors to *callback*.

        ``callback(layer_id, tensor_data, flags, addr) -> None``
        """
        self.sock.settimeout(timeout)
        self._recv_running = True
        logger.info(f"AITP recv_loop started on [::]:{self.port}")

        # FEC reassembly buffer: {(addr, layer_id, msg_id): {...}}
        fec_buf: dict = {}
        # messages déjà reconstruits : leurs fragments de parité tardifs sont ignorés
        # (sinon ils rouvriraient un réassemblage qui finirait en faux « perdu »)
        fec_done: dict = {}

        while self._recv_running:
            try:
                data, addr = self.sock.recvfrom(65535)
                parsed = self.parse_packet(data)

                if parsed["flags"] & FLAG_FEC and self._fec is not None:
                    # FEC shard — reassemble
                    tensor_data = parsed["tensor_data"]
                    lid = parsed["layer_id"]
                    if parsed["version"] >= 2:
                        hdr = struct.calcsize(FEC_META_V2)
                        if len(tensor_data) < hdr:
                            continue
                        total, idx, orig_size, msg_id = struct.unpack(
                            FEC_META_V2, tensor_data[:hdr])
                    else:  # compat v1 : pas d'identifiant, risque de mélange documenté
                        hdr = struct.calcsize(FEC_META_V1)
                        if len(tensor_data) < hdr:
                            continue
                        total, idx, orig_size = struct.unpack(FEC_META_V1, tensor_data[:hdr])
                        msg_id = -1
                    shard_data = tensor_data[hdr:]
                    key = (addr[0] if addr else None, lid, msg_id)

                    # purge des réassemblages incomplets trop vieux (pertes > parité) :
                    # sans ça le tampon grossit indéfiniment
                    now = time.monotonic()
                    for k in [k for k, v in fec_buf.items()
                              if now - v["t0"] > FEC_REASSEMBLY_TIMEOUT_S]:
                        del fec_buf[k]
                        if _AITP_ERRORS:
                            _AITP_ERRORS.labels("fec_timeout").inc()
                        logger.warning("AITP : tenseur couche=%s msg=%s perdu "
                                       "(trop de fragments manquants)", k[1], k[2])

                    for k in [k for k, t in fec_done.items()
                              if now - t > FEC_REASSEMBLY_TIMEOUT_S]:
                        del fec_done[k]
                    if key in fec_done:
                        continue
                    if key not in fec_buf:
                        fec_buf[key] = {"total": total, "orig": orig_size,
                                        "shards": {}, "t0": now}
                    fec_buf[key]["shards"][idx] = shard_data

                    # Try decode when enough shards collected
                    if len(fec_buf[key]["shards"]) >= self._fec.data_shards:
                        try:
                            reconstructed = self._fec.decode(
                                fec_buf[key]["shards"],
                                fec_buf[key]["orig"],
                            )
                            del fec_buf[key]
                            if msg_id >= 0:
                                fec_done[key] = now
                            if callback:
                                callback(lid, reconstructed, parsed["flags"], addr)
                        except Exception as e:
                            logger.debug(f"AITP FEC decode pending: {e}")
                else:
                    if callback:
                        callback(
                            parsed["layer_id"],
                            parsed["tensor_data"],
                            parsed["flags"],
                            addr,
                        )
            except socket.timeout:
                continue
            except ValueError as e:
                logger.debug(f"AITP bad packet: {e}")
                if _AITP_ERRORS:
                    _AITP_ERRORS.labels("parse_error").inc()
            except Exception as e:
                logger.debug(f"AITP recv error: {e}")

    def stop_recv(self):
        """Signal the recv_loop to stop."""
        self._recv_running = False


# ── Load-balanced send ─────────────────────────────────────────────────

    def send_balanced(self, layer_id: int, tensor_bytes: bytes, retries: int = 2) -> bool:
        """Send tensor via anycast load balancer with automatic failover.

        Uses the global AnycastLoadBalancer to pick the best healthy node
        (based on Connectome synapse weights), with retry on failure.
        Falls back to ``send_anycast(self.anycast_ipv6, ...)`` if no LB.
        """
        try:
            from core.network.anycast_balancer import get_anycast_balancer
            lb = get_anycast_balancer()
            return lb.select_and_send(self, layer_id, tensor_bytes, retries=retries)
        except ImportError:
            # Fallback: direct anycast send
            self.send_anycast(self.anycast_ipv6, layer_id, tensor_bytes)
            return True

    def send_raid(
        self,
        layer_id: int,
        tensor_bytes: bytes,
        data_shards: int = None,
        parity_shards: int = 2,
    ) -> bool:
        """Send tensor via Network RAID (striped across nodes + RS parity).

        Stripes the tensor into data_shards fragments, adds RS parity
        shards, and sends them in parallel to different cluster nodes.
        Receivers can reconstruct even if up to parity_shards nodes fail.

        Args:
            layer_id: Layer identifier for AITP packets.
            tensor_bytes: Raw tensor bytes to distribute.
            data_shards: Number of data stripes (None = auto from cluster).
            parity_shards: Number of RS parity stripes (default 2).

        Returns True if all shards were sent (some may fail tolerably).
        """
        try:
            from core.network.network_raid import NetworkRAID
            from core.network.anycast_balancer import get_anycast_balancer
            lb = get_anycast_balancer()
            raid = NetworkRAID(data_shards=data_shards, parity_shards=parity_shards)
            raid_id = raid.stripe_send(
                tensor_bytes, layer_id,
                aitp_protocol=self,
                balancer=lb,
            )
            return raid_id is not None
        except ImportError as e:
            logger.warning(f"AITP RAID unavailable: {e}, falling back to direct send")
            self.send_anycast(self.anycast_ipv6, layer_id, tensor_bytes)
            return True


# ── Singleton ──────────────────────────────────────────────────────────
_global_aitp = None

def get_aitp_protocol():
    global _global_aitp
    if _global_aitp is None:
        _global_aitp = AITPProtocol()
    return _global_aitp
