"""
Parity Memory — XOR Erasure Coding for Distributed KV Cache
============================================================

RAID-5-style redundancy for distributed tensors. Splits a tensor blob
into N data shards + 1 XOR parity shard. Can reconstruct any single
missing shard from the remaining shards + parity.

Single fault tolerance only (1 lost shard). For multi-fault tolerance,
use ``core.network.aitp_fec.FastFEC`` which implements real GF(2^8)
Cauchy Reed-Solomon (up to ``parity_shards`` simultaneous losses).

Uses Rust native (vramancer_rust) or C++ (swarm_core) SIMD-accelerated
XOR when available, falls back to pure Python.

Formerly named ``holographic_memory.py`` — renamed for honesty.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger("vramancer.parity_memory")

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

# ── Prometheus metrics (lazy) ──────────────────────────────────────────
_PARITY_ENCODES = None
_PARITY_HEALS = None
_PARITY_FAILURES = None


def _init_parity_metrics():
    global _PARITY_ENCODES, _PARITY_HEALS, _PARITY_FAILURES
    if _PARITY_ENCODES is not None:
        return
    try:
        from prometheus_client import Counter
        _PARITY_ENCODES = Counter(
            "vramancer_parity_encodes_total",
            "XOR parity encode operations",
        )
        _PARITY_HEALS = Counter(
            "vramancer_parity_heals_total",
            "XOR parity shard reconstructions",
        )
        _PARITY_FAILURES = Counter(
            "vramancer_parity_failures_total",
            "Parity reconstruction failures (>1 shard lost)",
        )
    except Exception:
        pass


class ParityKVManager:
    """XOR parity erasure coding for distributed tensor shards.

    Provides single-fault-tolerance: can recover from exactly 1 lost
    shard out of N. For stronger guarantees, use ``FastFEC`` which
    provides Reed-Solomon with configurable redundancy.
    """

    def __init__(self):
        self.active_engrams: Dict[str, Dict[str, Any]] = {}
        self.native_core = None
        self._use_native = False

        _init_parity_metrics()

        if _MINIMAL:
            return

        try:
            import vramancer_rust as _native
            self.native_core = _native
            self._use_native = True
            logger.info("Parity: using Rust native XOR (SIMD)")
        except ImportError:
            try:
                import swarm_core as _native
                self.native_core = _native
                self._use_native = True
                logger.info("Parity: using C++ native XOR (SIMD)")
            except ImportError:
                logger.debug("Parity: using Python fallback XOR")

    def _xor_bytes(self, b1: bytes, b2: bytes) -> bytes:
        """XOR two equal-length byte strings."""
        return bytes(a ^ b for a, b in zip(b1, b2))

    def encode(
        self, tensor_blob: bytes, num_shards: int,
    ) -> Tuple[List[bytes], bytes]:
        """Split tensor into *num_shards* data shards + 1 XOR parity shard."""
        shard_size = len(tensor_blob) // num_shards
        shards = []

        for i in range(num_shards):
            start = i * shard_size
            end = start + shard_size if i < num_shards - 1 else len(tensor_blob)
            shards.append(tensor_blob[start:end])

        # Pad to equal length for XOR
        max_len = max(len(s) for s in shards)
        padded_shards = [s.ljust(max_len, b'\x00') for s in shards]

        # Generate parity
        if self._use_native and self.native_core is not None:
            parity = self.native_core.generate_holographic_parity(padded_shards)
        else:
            parity = bytearray(max_len)
            for shard in padded_shards:
                parity = self._xor_bytes(parity, shard)
            parity = bytes(parity)

        if _PARITY_ENCODES:
            _PARITY_ENCODES.inc()

        return padded_shards, parity

    # Backward-compat alias
    encode_hologram = encode

    def heal(
        self, shards: List[Optional[bytes]], parity: bytes,
    ) -> bytes:
        """Reconstruct a single missing shard using XOR parity.

        Returns concatenated data, or ``b""`` if more than one shard is
        missing (XOR parity can only recover 1).
        """
        missing_index = -1
        for i, shard in enumerate(shards):
            if shard is None:
                if missing_index != -1:
                    logger.error("Parity: multiple shards lost, cannot heal")
                    if _PARITY_FAILURES:
                        _PARITY_FAILURES.inc()
                    return b""
                missing_index = i

        if missing_index == -1:
            return b"".join(shards)

        logger.warning(f"Parity: healing shard {missing_index}")

        if self._use_native and self.native_core is not None:
            valid_shards = [
                s for i, s in enumerate(shards)
                if i != missing_index and s is not None
            ]
            reconstructed_shard = self.native_core.heal_holograph(
                valid_shards, parity,
            )
        else:
            reconstructed_shard = bytearray(len(parity))
            for i, shard in enumerate(shards):
                if i != missing_index and shard is not None:
                    reconstructed_shard = self._xor_bytes(
                        reconstructed_shard, shard,
                    )
            reconstructed_shard = self._xor_bytes(reconstructed_shard, parity)
            reconstructed_shard = bytes(reconstructed_shard)

        shards[missing_index] = reconstructed_shard

        if _PARITY_HEALS:
            _PARITY_HEALS.inc()
        logger.info("Parity: shard restored")
        return b"".join(shards)

    # Backward-compat alias
    heal_hologram = heal

    def store_engram(
        self, engram_id: str, tensor_blob: bytes, num_shards: int,
    ) -> Dict[str, Any]:
        """Encode and store a tensor with parity for later healing."""
        padded_shards, parity = self.encode(tensor_blob, num_shards)
        self.active_engrams[engram_id] = {
            "shards": padded_shards,
            "parity": parity,
            "original_size": len(tensor_blob),
            "num_shards": num_shards,
            "created_at": time.time(),
        }
        return {"engram_id": engram_id, "num_shards": num_shards, "parity_bytes": len(parity)}

    def heal_engram(self, engram_id: str, missing_idx: int) -> Optional[bytes]:
        """Simulate loss of shard *missing_idx* and heal it."""
        engram = self.active_engrams.get(engram_id)
        if engram is None:
            return None
        shards = list(engram["shards"])
        shards[missing_idx] = None
        result = self.heal(shards, engram["parity"])
        if result:
            engram["shards"] = shards  # updated with healed shard
        return result or None

    def stats(self) -> Dict[str, Any]:
        return {
            "active_engrams": len(self.active_engrams),
            "native": self._use_native,
        }


# Backward-compat alias
HolographicKVManager = ParityKVManager

# Singleton
parity_kv = ParityKVManager()
hive_memory = parity_kv  # backward-compat
