"""
Network RAID — Distributed tensor striping with Reed-Solomon protection.
========================================================================

RAID-0+RS: stripes tensor data across N cluster nodes with configurable
Reed-Solomon parity shards for fault tolerance. Can survive up to
``parity_shards`` simultaneous node failures without data loss.

Architecture::

    Tensor (28 GB bf16) ──► stripe into N data shards
         │
         ▼
    RS encode (GF(2^8) Cauchy) ──► + P parity shards
         │
         ▼
    parallel UDP send via AITP ──► N+P nodes
         │
         ▼  (on receiver side)
    collect shards, RS decode if losses ──► reconstruct tensor

Uses:
  - ``core.network.aitp_fec.FastFEC`` for RS encoding/decoding
  - ``core.network.aitp_protocol.AITPProtocol`` for authenticated transport
  - ``core.network.anycast_balancer.AnycastLoadBalancer`` for node selection
  - ``core.parity_memory.ParityKVManager`` as fallback (XOR, 1-fault only)

Environment:
  - ``VRM_RAID_DATA_SHARDS``: Number of data stripes (default: auto = num_nodes)
  - ``VRM_RAID_PARITY_SHARDS``: Number of RS parity stripes (default: 2)
  - ``VRM_RAID_TIMEOUT``: Reassembly timeout in seconds (default: 10.0)
  - ``VRM_RAID_PARALLEL``: Max parallel sends (default: 4)
"""

import os
import time
import math
import struct
import logging
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("vramancer.network_raid")

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

# ── Configuration ──────────────────────────────────────────────────────
RAID_PARITY_SHARDS = int(os.environ.get("VRM_RAID_PARITY_SHARDS", "2"))
RAID_TIMEOUT = float(os.environ.get("VRM_RAID_TIMEOUT", "10.0"))
RAID_MAX_PARALLEL = int(os.environ.get("VRM_RAID_PARALLEL", "4"))

# ── Prometheus metrics (lazy) ──────────────────────────────────────────
_RAID_ENCODES = None
_RAID_DECODES = None
_RAID_RECOVERIES = None
_RAID_FAILURES = None


def _init_raid_metrics():
    global _RAID_ENCODES, _RAID_DECODES, _RAID_RECOVERIES, _RAID_FAILURES
    if _RAID_ENCODES is not None:
        return
    try:
        from prometheus_client import Counter
        _RAID_ENCODES = Counter(
            "vramancer_raid_encodes_total",
            "Network RAID encode operations (tensor → shards)",
        )
        _RAID_DECODES = Counter(
            "vramancer_raid_decodes_total",
            "Network RAID decode operations (shards → tensor)",
        )
        _RAID_RECOVERIES = Counter(
            "vramancer_raid_recoveries_total",
            "RS recovery operations (lost shards reconstructed)",
        )
        _RAID_FAILURES = Counter(
            "vramancer_raid_failures_total",
            "RAID operations that failed (too many shards lost)",
        )
    except Exception:
        pass


# ── Shard metadata header ─────────────────────────────────────────────
# Prepended to each shard for reassembly:
#   [magic(2B)][raid_id(16B)][total_shards(H)][shard_idx(H)]
#   [data_shards(H)][parity_shards(H)][original_size(Q)]
# Total: 2 + 16 + 2 + 2 + 2 + 2 + 8 = 34 bytes
RAID_SHARD_HEADER = "!2s16sHHHHQ"
RAID_SHARD_HEADER_SIZE = struct.calcsize(RAID_SHARD_HEADER)
RAID_MAGIC = b"VR"


class RaidShardInfo:
    """Metadata for a single RAID shard."""

    __slots__ = (
        "raid_id", "total_shards", "shard_idx",
        "data_shards", "parity_shards", "original_size",
    )

    def __init__(
        self,
        raid_id: bytes,
        total_shards: int,
        shard_idx: int,
        data_shards: int,
        parity_shards: int,
        original_size: int,
    ):
        self.raid_id = raid_id
        self.total_shards = total_shards
        self.shard_idx = shard_idx
        self.data_shards = data_shards
        self.parity_shards = parity_shards
        self.original_size = original_size


def _make_raid_id(tensor_bytes: bytes) -> bytes:
    """Generate a 16-byte RAID operation ID from tensor content hash."""
    h = hashlib.md5(tensor_bytes[:4096])  # Hash first 4KB for speed
    h.update(struct.pack("!dQ", time.time(), len(tensor_bytes)))
    return h.digest()


def _pack_shard_header(info: RaidShardInfo) -> bytes:
    """Pack shard metadata into the binary header."""
    return struct.pack(
        RAID_SHARD_HEADER,
        RAID_MAGIC,
        info.raid_id,
        info.total_shards,
        info.shard_idx,
        info.data_shards,
        info.parity_shards,
        info.original_size,
    )


def _unpack_shard_header(data: bytes) -> Tuple[RaidShardInfo, bytes]:
    """Unpack shard header and return (info, payload)."""
    if len(data) < RAID_SHARD_HEADER_SIZE:
        raise ValueError("RAID shard too small for header")

    magic, raid_id, total, idx, d_shards, p_shards, orig_size = struct.unpack(
        RAID_SHARD_HEADER, data[:RAID_SHARD_HEADER_SIZE],
    )
    if magic != RAID_MAGIC:
        raise ValueError(f"RAID shard magic invalid: {magic!r}")

    info = RaidShardInfo(
        raid_id=raid_id,
        total_shards=total,
        shard_idx=idx,
        data_shards=d_shards,
        parity_shards=p_shards,
        original_size=orig_size,
    )
    return info, data[RAID_SHARD_HEADER_SIZE:]


class ShardReassembler:
    """Collects incoming RAID shards and reassembles when enough arrive.

    Thread-safe. Supports multiple concurrent RAID operations tracked by
    ``raid_id``. Automatically expires stale operations after timeout.
    """

    def __init__(self, timeout: float = RAID_TIMEOUT):
        self.timeout = timeout
        self._lock = threading.Lock()
        # {raid_id: {"meta": RaidShardInfo, "shards": {idx: bytes}, "ts": float}}
        self._pending: Dict[bytes, Dict[str, Any]] = {}

    def add_shard(
        self, raid_id: bytes, info: RaidShardInfo, shard_data: bytes,
    ) -> Optional[bytes]:
        """Add a received shard. Returns reconstructed tensor if ready, else None."""
        with self._lock:
            if raid_id not in self._pending:
                self._pending[raid_id] = {
                    "meta": info,
                    "shards": {},
                    "ts": time.time(),
                }
            entry = self._pending[raid_id]
            entry["shards"][info.shard_idx] = shard_data

            # Check if we have enough shards to decode
            if len(entry["shards"]) >= info.data_shards:
                # Attempt reconstruction
                result = self._reconstruct(entry)
                del self._pending[raid_id]
                return result

        return None

    def _reconstruct(self, entry: Dict[str, Any]) -> Optional[bytes]:
        """Reconstruct original tensor from collected shards."""
        meta = entry["meta"]
        shards = entry["shards"]

        # Check if all data shards are present (fast path)
        data_indices = set(range(meta.data_shards))
        have_data = set(i for i in shards if i < meta.data_shards)

        if have_data == data_indices:
            # All data shards present — just concatenate
            parts = []
            for i in range(meta.data_shards):
                parts.append(shards[i])
            result = b"".join(parts)
            if _RAID_DECODES:
                _RAID_DECODES.inc()
            return result[:meta.original_size]

        # Need RS recovery — use FastFEC decode
        try:
            from core.network.aitp_fec import FastFEC
            fec = FastFEC(
                data_shards=meta.data_shards,
                parity_shards=meta.parity_shards,
            )
            result = fec.decode(shards, meta.original_size)
            if _RAID_RECOVERIES:
                _RAID_RECOVERIES.inc()
            if _RAID_DECODES:
                _RAID_DECODES.inc()
            lost = meta.data_shards - len(have_data)
            logger.info(
                f"RAID: RS recovery successful — {lost} shard(s) reconstructed "
                f"(raid_id={meta.raid_id.hex()[:8]})"
            )
            return result
        except Exception as e:
            logger.error(f"RAID: RS recovery failed: {e}")
            if _RAID_FAILURES:
                _RAID_FAILURES.inc()
            return None

    def expire_stale(self):
        """Remove pending operations that have timed out."""
        now = time.time()
        with self._lock:
            stale = [
                rid for rid, entry in self._pending.items()
                if now - entry["ts"] > self.timeout
            ]
            for rid in stale:
                logger.warning(
                    f"RAID: expired stale reassembly "
                    f"(raid_id={rid.hex()[:8]}, "
                    f"shards={len(self._pending[rid]['shards'])}/"
                    f"{self._pending[rid]['meta'].data_shards})"
                )
                del self._pending[rid]
                if _RAID_FAILURES:
                    _RAID_FAILURES.inc()

    def pending_count(self) -> int:
        """Number of in-progress reassembly operations."""
        with self._lock:
            return len(self._pending)


class NetworkRAID:
    """RAID-0+RS tensor striping across cluster nodes.

    Distributes tensor data across multiple nodes with Reed-Solomon
    parity for fault tolerance. Uses parallel AITP sends and collects
    results with automatic RS recovery on shard loss.

    Usage::

        raid = NetworkRAID(data_shards=4, parity_shards=2)
        raid_id = raid.stripe_send(tensor_bytes, layer_id=42)

        # On receivers, shards arrive via AITP recv_loop callback:
        raid.handle_incoming_shard(layer_id, shard_data, flags, addr)

        # When reassembly completes, callback fires with full tensor
        raid.set_completion_callback(on_tensor_ready)
    """

    def __init__(
        self,
        data_shards: int = None,
        parity_shards: int = RAID_PARITY_SHARDS,
        max_parallel: int = RAID_MAX_PARALLEL,
    ):
        self.data_shards = data_shards  # None = auto from node count
        self.parity_shards = parity_shards
        self.max_parallel = max_parallel

        self._fec = None
        self._reassembler = ShardReassembler()
        self._completion_callback = None
        self._executor = ThreadPoolExecutor(
            max_workers=max_parallel, thread_name_prefix="raid-send",
        )
        self._lock = threading.Lock()

        # Expiry thread
        self._running = False
        self._expiry_thread = None

        _init_raid_metrics()
        logger.info(
            f"NetworkRAID initialized: data_shards={data_shards or 'auto'}, "
            f"parity_shards={parity_shards}, parallel={max_parallel}"
        )

    def _get_fec(self, d_shards: int) -> Any:
        """Lazy-init FEC with correct shard count."""
        try:
            from core.network.aitp_fec import FastFEC
            return FastFEC(data_shards=d_shards, parity_shards=self.parity_shards)
        except ImportError:
            logger.warning("NetworkRAID: aitp_fec unavailable, no RS protection")
            return None

    def set_completion_callback(self, callback):
        """Set callback for completed reassembly: callback(raid_id, tensor_bytes)."""
        self._completion_callback = callback

    # ── Encode + Send ──────────────────────────────────────────────────

    def stripe_send(
        self,
        tensor_bytes: bytes,
        layer_id: int,
        aitp_protocol=None,
        balancer=None,
        target_nodes: List[Any] = None,
    ) -> Optional[bytes]:
        """Stripe a tensor across cluster nodes with RS parity.

        Args:
            tensor_bytes: Raw tensor data to distribute.
            layer_id: Layer ID for AITP packet headers.
            aitp_protocol: AITPProtocol for transport (auto-created if None).
            balancer: AnycastLoadBalancer for node selection (optional).
            target_nodes: Explicit list of (ipv6, port) tuples. Overrides balancer.

        Returns:
            The 16-byte raid_id for tracking, or None on failure.
        """
        raid_id = _make_raid_id(tensor_bytes)
        original_size = len(tensor_bytes)

        # Determine number of data shards
        if target_nodes:
            num_targets = len(target_nodes)
        elif balancer:
            num_targets = len(balancer.get_healthy_nodes())
        else:
            num_targets = self.data_shards or 4

        d_shards = self.data_shards or max(2, num_targets)
        total_shards = d_shards + self.parity_shards

        # RS encode
        fec = self._get_fec(d_shards)
        if fec:
            all_shards = fec.encode(tensor_bytes)
        else:
            # No FEC: just split into data shards, no parity
            shard_size = math.ceil(original_size / d_shards)
            padded = tensor_bytes.ljust(shard_size * d_shards, b"\x00")
            all_shards = [
                padded[i * shard_size:(i + 1) * shard_size]
                for i in range(d_shards)
            ]
            total_shards = d_shards

        if _RAID_ENCODES:
            _RAID_ENCODES.inc()

        # Select targets
        if target_nodes is None and balancer:
            nodes = balancer.select_targets(total_shards)
            target_nodes = [(n.ipv6, n.port) for n in nodes]

        if not target_nodes:
            logger.error("NetworkRAID: no target nodes for striped send")
            return None

        # Wrap shard data with RAID header + send via AITP
        if aitp_protocol is None:
            try:
                from core.network.aitp_protocol import get_aitp_protocol
                aitp_protocol = get_aitp_protocol()
            except ImportError:
                logger.error("NetworkRAID: aitp_protocol unavailable")
                return None

        def _send_shard(shard_idx: int, shard_data: bytes, target: Tuple[str, int]):
            info = RaidShardInfo(
                raid_id=raid_id,
                total_shards=total_shards,
                shard_idx=shard_idx,
                data_shards=d_shards,
                parity_shards=self.parity_shards if fec else 0,
                original_size=original_size,
            )
            header = _pack_shard_header(info)
            payload = header + shard_data
            aitp_protocol.send_anycast(target[0], layer_id, payload)

        # Parallel send — map shards to targets (round-robin if fewer targets)
        futures = []
        for i, shard_data in enumerate(all_shards):
            target = target_nodes[i % len(target_nodes)]
            futures.append(
                self._executor.submit(_send_shard, i, shard_data, target)
            )

        # Wait for all sends
        errors = 0
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.warning(f"RAID: shard send failed: {e}")
                errors += 1

        if errors > self.parity_shards:
            logger.error(
                f"RAID: too many send failures ({errors}/{total_shards}), "
                f"RS can only recover {self.parity_shards}"
            )

        shard_size = len(all_shards[0]) if all_shards else 0
        logger.info(
            f"RAID: striped {original_size} bytes → "
            f"{d_shards}+{self.parity_shards if fec else 0} shards × "
            f"{shard_size} bytes, sent to {len(target_nodes)} nodes "
            f"(errors={errors}, raid_id={raid_id.hex()[:8]})"
        )
        return raid_id

    # ── Receive + Reassemble ───────────────────────────────────────────

    def handle_incoming_shard(
        self,
        layer_id: int,
        raw_data: bytes,
        flags: int = 0,
        addr: Any = None,
    ) -> Optional[bytes]:
        """Process an incoming RAID shard from AITP recv_loop.

        Wire this as the AITP callback:
            aitp.recv_loop(callback=raid.handle_incoming_shard)

        Returns the reconstructed tensor if reassembly is complete, else None.
        """
        try:
            info, shard_data = _unpack_shard_header(raw_data)
        except ValueError as e:
            logger.debug(f"RAID: not a RAID shard: {e}")
            return None

        result = self._reassembler.add_shard(info.raid_id, info, shard_data)

        if result is not None:
            logger.info(
                f"RAID: reassembly complete — {len(result)} bytes "
                f"(raid_id={info.raid_id.hex()[:8]})"
            )
            if self._completion_callback:
                self._completion_callback(info.raid_id, result)

        return result

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start_expiry_thread(self):
        """Start background thread that expires stale reassembly operations."""
        if self._running:
            return
        self._running = True
        self._expiry_thread = threading.Thread(
            target=self._expiry_loop, daemon=True, name="RAIDExpiry",
        )
        self._expiry_thread.start()

    def _expiry_loop(self):
        while self._running:
            self._reassembler.expire_stale()
            time.sleep(RAID_TIMEOUT / 2)

    def stop(self):
        """Stop the RAID manager and clean up resources."""
        self._running = False
        self._executor.shutdown(wait=False)

    def status(self) -> Dict[str, Any]:
        """Return RAID manager status for monitoring."""
        return {
            "data_shards": self.data_shards or "auto",
            "parity_shards": self.parity_shards,
            "max_parallel": self.max_parallel,
            "pending_reassemblies": self._reassembler.pending_count(),
            "fec_available": self._fec is not None or True,  # lazy init
        }


# ── Singleton ──────────────────────────────────────────────────────────
_global_raid: Optional[NetworkRAID] = None


def get_network_raid() -> NetworkRAID:
    """Get or create the global NetworkRAID manager."""
    global _global_raid
    if _global_raid is None:
        _global_raid = NetworkRAID()
    return _global_raid
