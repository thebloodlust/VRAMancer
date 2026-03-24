"""
Holographic Memory (XOR Parity Erasure Coding for KV Cache)
===========================================================

RAID-5-style redundancy for distributed tensors. Splits a tensor blob
into N data shards + 1 XOR parity shard. Can reconstruct any single
missing shard from the remaining shards + parity.

Uses Rust native (vramancer_rust) or C++ (swarm_core) SIMD-accelerated
XOR when available, falls back to pure Python.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger("vramancer.holographic_memory")

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

# ── Prometheus metrics (lazy) ──────────────────────────────────────────
_HOLO_ENCODES = None
_HOLO_HEALS = None
_HOLO_FAILURES = None

def _init_holo_metrics():
    global _HOLO_ENCODES, _HOLO_HEALS, _HOLO_FAILURES
    if _HOLO_ENCODES is not None:
        return
    try:
        from prometheus_client import Counter
        _HOLO_ENCODES = Counter(
            "vramancer_holographic_encodes_total",
            "Holographic parity encode operations",
        )
        _HOLO_HEALS = Counter(
            "vramancer_holographic_heals_total",
            "Holographic shard reconstructions",
        )
        _HOLO_FAILURES = Counter(
            "vramancer_holographic_failures_total",
            "Holographic reconstruction failures (>1 shard lost)",
        )
    except Exception:
        pass


class HolographicKVManager:
    """XOR parity erasure coding for distributed tensor shards."""

    def __init__(self):
        self.active_engrams: Dict[str, Dict[str, Any]] = {}
        self.native_core = None
        self._use_native = False

        _init_holo_metrics()

        if _MINIMAL:
            return

        try:
            import vramancer_rust as swarm_core
            self.native_core = swarm_core
            self._use_native = True
            logger.info("Holographic: using Rust native XOR (SIMD)")
        except ImportError:
            try:
                import swarm_core
                self.native_core = swarm_core
                self._use_native = True
                logger.info("Holographic: using C++ native XOR (SIMD)")
            except ImportError:
                logger.debug("Holographic: using Python fallback XOR")

    def _xor_bytes(self, b1: bytes, b2: bytes) -> bytes:
        """XOR two equal-length byte strings."""
        return bytes(a ^ b for a, b in zip(b1, b2))

    def encode_hologram(
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

        if _HOLO_ENCODES:
            _HOLO_ENCODES.inc()

        return padded_shards, parity

    def heal_hologram(
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
                    logger.error("Holographic: multiple shards lost, cannot heal")
                    if _HOLO_FAILURES:
                        _HOLO_FAILURES.inc()
                    return b""
                missing_index = i

        if missing_index == -1:
            return b"".join(shards)

        logger.warning(f"Holographic: healing shard {missing_index}")

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

        if _HOLO_HEALS:
            _HOLO_HEALS.inc()
        logger.info("Holographic: shard restored")
        return b"".join(shards)

    def store_engram(
        self, engram_id: str, tensor_blob: bytes, num_shards: int,
    ) -> Dict[str, Any]:
        """Encode and store a tensor with parity for later healing."""
        padded_shards, parity = self.encode_hologram(tensor_blob, num_shards)
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
        result = self.heal_hologram(shards, engram["parity"])
        if result:
            engram["shards"] = shards  # updated with healed shard
        return result or None

    def stats(self) -> Dict[str, Any]:
        return {
            "active_engrams": len(self.active_engrams),
            "native": self._use_native,
        }


# Singleton
hive_memory = HolographicKVManager()
