"""Unified transport factory for VRAMancer.

Selects the optimal transport based on locality:

  Same GPU   -> no-op (direct pointer)
  Same node  -> NCCL P2P / CUDA copy via TransferManager
  Same rack  -> VTP (LLM Transport Protocol) with GPUDirect RDMA
  WAN        -> VTP TCP fallback / Zero-copy TCP

Integrates the new LLM Transport Protocol (VTP) for multi-node
inference with GPUDirect RDMA support.

Usage:
    factory = TransportFactory()
    transport = factory.get_transport(source, target)
    transport.transfer(tensor)
"""
from __future__ import annotations

import os
import time
import threading
from enum import Enum, auto
from typing import Any, Optional, Dict
from dataclasses import dataclass

from core.logger import LoggerAdapter

log = LoggerAdapter("transport")

# --- Conditional imports ---
try:
    from core.metrics import FASTPATH_BYTES, FASTPATH_LATENCY, GPU_TRANSFER_OPS
    _METRICS = True
except Exception:
    _METRICS = False

# Lazy imports to avoid circular dependencies
_transfer_manager = None
_fastpath_channel = None
_vtp_transport = None
_lock = threading.Lock()


class Locality(Enum):
    """Where source and target are relative to each other."""
    SAME_GPU = auto()       # tensor already on target device
    SAME_NODE = auto()      # different GPU, same machine (PCIe/NVLink)
    SAME_RACK = auto()      # different machine, fast network (RDMA/25GbE+)
    REMOTE = auto()         # WAN / internet / slow link


@dataclass
class TransportTarget:
    """Describes a transfer endpoint."""
    node_id: str = "local"       # hostname or node identifier
    gpu_id: int = 0              # GPU device index on that node
    is_local: bool = True        # True if same machine as us

    @property
    def device_str(self) -> str:
        return f"cuda:{self.gpu_id}" if self.is_local else f"{self.node_id}:cuda:{self.gpu_id}"


class TransportFactory:
    """Selects and manages the optimal transport for each source/target pair.

    Architecture:
        L1 <-> L2 (VRAM inter-GPU):  TransferManager (NCCL / P2P / CPU-staged)
        L2 <-> L3 (GPU <-> RAM):     torch.Tensor.cpu() / .cuda()
        L3 <-> L4 (RAM <-> remote):  VTP (LLM Transport Protocol) / FastHandle
        L3 <-> L5 (RAM <-> NVMe):    pickle / mmap (handled by HierarchicalMemoryManager)
        L4 <-> L6 (remote network):  VTP TCP fallback

    Production features:
        - Thread-safe singleton management
        - Prometheus metrics on all transfers
        - Automatic VTP integration for multi-node LLM inference
        - Transfer statistics and diagnostics
    """

    def __init__(self):
        self._local_node_id = os.environ.get("VRM_NODE_ID", "local")
        self._transfer_count = 0
        self._total_bytes = 0
        self._errors = 0
        self._lock = threading.Lock()

    def determine_locality(
        self,
        source: TransportTarget,
        target: TransportTarget,
    ) -> Locality:
        """Determine the locality relationship between source and target."""
        if source.node_id != target.node_id:
            # Different nodes - check if same rack (heuristic)
            same_rack = os.environ.get("VRM_SAME_RACK_NODES", "")
            if target.node_id in same_rack.split(","):
                return Locality.SAME_RACK
            return Locality.REMOTE

        # Same node
        if source.gpu_id == target.gpu_id:
            return Locality.SAME_GPU
        return Locality.SAME_NODE

    def get_local_transfer_manager(self):
        """Get the singleton TransferManager for local GPU-to-GPU transfers."""
        global _transfer_manager
        if _transfer_manager is None:
            with _lock:
                if _transfer_manager is None:
                    from core.transfer_manager import TransferManager
                    _transfer_manager = TransferManager(verbose=True)
        return _transfer_manager

    def get_network_channel(self, remote_host: Optional[str] = None, remote_port: int = 18900):
        """Get the singleton FastHandle for network transfers."""
        global _fastpath_channel
        if _fastpath_channel is None:
            with _lock:
                if _fastpath_channel is None:
                    from core.network.fibre_fastpath import open_low_latency_channel
                    _fastpath_channel = open_low_latency_channel(
                        remote_host=remote_host,
                        remote_port=remote_port,
                    )
        return _fastpath_channel

    def get_vtp_transport(self, node_id: Optional[str] = None):
        """Get the VTP (LLM Transport Protocol) for multi-node transfers.

        VTP provides:
          - GPUDirect RDMA (NIC ↔ GPU VRAM, zero CPU copy)
          - Tensor-aware framing (shape/dtype in header)
          - Pipeline overlap with double-buffered regions
          - KV cache streaming protocol
        """
        global _vtp_transport
        if _vtp_transport is None:
            with _lock:
                if _vtp_transport is None:
                    try:
                        from core.network.llm_transport import LLMTransport
                        nid = node_id or self._local_node_id
                        _vtp_transport = LLMTransport(node_id=nid)
                        log.info(f"VTP transport initialized (tier={_vtp_transport.tier.name})")
                    except Exception as exc:
                        log.warning(f"VTP init failed: {exc} — using FastHandle fallback")
                        return None
        return _vtp_transport

    def transfer_tensor(
        self,
        tensor: Any,
        source: TransportTarget,
        target: TransportTarget,
    ) -> Dict[str, Any]:
        """Transfer a tensor using the optimal transport for the given locality.

        Returns a dict with transfer metadata (method, duration, bandwidth, etc).
        """
        locality = self.determine_locality(source, target)

        if locality == Locality.SAME_GPU:
            return {
                "method": "no-op",
                "locality": "same_gpu",
                "bytes": 0,
                "duration_s": 0.0,
            }

        if locality == Locality.SAME_NODE:
            # Use NCCL / P2P via TransferManager
            try:
                tm = self.get_local_transfer_manager()
                result = tm.send_activation(source.gpu_id, target.gpu_id, tensor)
                meta = {
                    "method": result.method.name,
                    "locality": "same_node",
                    "bytes": result.bytes_transferred,
                    "duration_s": result.duration_s,
                    "bandwidth_gbps": result.bandwidth_gbps,
                }
                self._record_transfer(result.bytes_transferred)
                if _METRICS:
                    GPU_TRANSFER_OPS.labels(
                        result.method.name, str(source.gpu_id), str(target.gpu_id)
                    ).inc()
                return meta
            except Exception as exc:
                self._errors += 1
                log.error(f"Local transfer {source} -> {target} failed: {exc}")
                raise

        if locality in (Locality.SAME_RACK, Locality.REMOTE):
            # Try VTP (LLM Transport Protocol) first — GPUDirect RDMA
            vtp = self.get_vtp_transport()
            if vtp is not None:
                try:
                    result = vtp.send_tensor(
                        tensor, dst_node=target.node_id,
                        dst_gpu=target.gpu_id,
                    )
                    self._record_transfer(result.get("bytes", 0))
                    result["locality"] = locality.name.lower()
                    return result
                except Exception as exc:
                    log.warning(f"VTP send failed, falling back to FastHandle: {exc}")

            # Fallback: FastHandle (RDMA / TCP)
            try:
                channel = self.get_network_channel(
                    remote_host=target.node_id if not target.is_local else None,
                )
                if channel is None:
                    raise RuntimeError(f"No transport available for {source} -> {target}")

                sent = channel.send_tensor(tensor, source.gpu_id)
                self._record_transfer(sent)
                return {
                    "method": channel.kind,
                    "locality": locality.name.lower(),
                    "bytes": sent,
                    "transport": channel.capabilities(),
                }
            except Exception as exc:
                self._errors += 1
                log.error(f"Network transfer {source} -> {target} failed: {exc}")
                raise

        raise RuntimeError(f"Unknown locality: {locality}")

    def _record_transfer(self, nbytes: int):
        """Record transfer stats."""
        self._transfer_count += 1
        self._total_bytes += nbytes

    def summary(self) -> Dict[str, Any]:
        """Return transport factory status."""
        local_tm = None
        try:
            if _transfer_manager:
                local_tm = _transfer_manager.stats()
        except Exception:
            pass

        net_caps = None
        try:
            if _fastpath_channel:
                net_caps = _fastpath_channel.capabilities()
        except Exception:
            pass

        vtp_stats = None
        try:
            if _vtp_transport:
                vtp_stats = _vtp_transport.stats()
        except Exception:
            pass

        return {
            "node_id": self._local_node_id,
            "local_transfer": local_tm,
            "network_channel": net_caps,
            "vtp": vtp_stats,
            "gpudirect_available": vtp_stats.get("gpudirect_available", False) if vtp_stats else False,
            "total_transfers": self._transfer_count,
            "total_bytes": self._total_bytes,
            "errors": self._errors,
        }


# Singleton
_factory: Optional[TransportFactory] = None

def get_transport_factory() -> TransportFactory:
    """Get the global TransportFactory singleton (thread-safe)."""
    global _factory
    if _factory is None:
        with _lock:
            if _factory is None:
                _factory = TransportFactory()
    return _factory


def reset_transport_factory():
    """Reset the global transport factory and all managed transports."""
    global _factory, _transfer_manager, _fastpath_channel, _vtp_transport
    if _vtp_transport:
        try:
            _vtp_transport.close()
        except Exception:
            pass
    _factory = None
    _transfer_manager = None
    _fastpath_channel = None
    _vtp_transport = None


__all__ = [
    "TransportFactory",
    "TransportTarget",
    "Locality",
    "get_transport_factory",
    "reset_transport_factory",
]
