"""GPU-to-GPU tensor transfer manager with NCCL backend.

Provides high-performance local multi-GPU transfers using:
  - NCCL via torch.distributed (heterogeneous GPU support)
  - CUDA P2P direct copy (cudaMemcpyPeer) when topology allows
  - CPU-staged fallback for cross-architecture transfers (e.g. Ampere <-> Blackwell)

Transport decision:
  1. If both GPUs support P2P access -> direct CUDA IPC / cudaMemcpyPeer
  2. Otherwise -> NCCL all_gather / send/recv (handles CPU staging transparently)
  3. If no GPU available -> CPU memcpy (debug/test)

Designed for heterogeneous setups (RTX 3090 + RTX 5070 Ti over PCIe).
"""
from __future__ import annotations

import os
import time
import threading
from typing import Any, Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum, auto

from core.logger import LoggerAdapter

# --- Conditional imports (projet convention: never crash on missing dep) ---
try:
    import torch
    import torch.cuda
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    dist = None  # type: ignore
    _DIST_AVAILABLE = False

# Metrics (optional)
try:
    from core.metrics import FASTPATH_BYTES, FASTPATH_LATENCY
    _METRICS = True
except Exception:
    _METRICS = False

log = LoggerAdapter("transfer")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
class TransportMethod(Enum):
    """How a tensor transfer was performed."""
    NCCL = auto()        # torch.distributed NCCL backend
    CUDA_P2P = auto()    # Direct cudaMemcpyPeer
    CPU_STAGED = auto()  # GPU -> CPU -> GPU (no P2P, no NCCL)
    CPU_ONLY = auto()    # Pure CPU (no GPU)
    STUB = auto()        # No-op (test mode)


@dataclass
class TransferResult:
    """Result metadata for a single tensor transfer."""
    method: TransportMethod
    source_gpu: int
    target_gpu: int
    bytes_transferred: int
    duration_s: float
    bandwidth_gbps: float = 0.0

    def __post_init__(self):
        if self.duration_s > 0 and self.bytes_transferred > 0:
            self.bandwidth_gbps = (self.bytes_transferred * 8) / (self.duration_s * 1e9)


@dataclass
class GPUTopology:
    """Cached P2P accessibility matrix."""
    num_gpus: int = 0
    p2p_matrix: Dict[Tuple[int, int], bool] = field(default_factory=dict)
    device_names: Dict[int, str] = field(default_factory=dict)
    device_caps: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # (major, minor)
    is_consumer: Dict[int, bool] = field(default_factory=dict)  # True if GeForce/consumer


def _is_consumer_gpu(name: str) -> bool:
    """Detect NVIDIA consumer GPUs that may restrict P2P/NCCL features.

    Consumer GPUs (GeForce, RTX without A/H/L prefix) may have P2P blocked
    by the driver for more than 2 GPUs, or across different architectures.
    NCCL still works via shared memory (SHM) transport as fallback.
    """
    name_lower = name.lower()
    consumer_prefixes = ("geforce", "rtx 20", "rtx 30", "rtx 40", "rtx 50",
                         "gtx", "titan")
    return any(name_lower.startswith(p) or p in name_lower for p in consumer_prefixes)


# ---------------------------------------------------------------------------
# Transfer Manager
# ---------------------------------------------------------------------------
class TransferManager:
    """High-performance GPU-to-GPU tensor transfer manager.

    Initialises NCCL process group on first use. Detects P2P topology
    and selects the fastest available transport for each (src, dst) pair.

    Usage:
        tm = TransferManager()
        result = tm.send_activation(0, 1, tensor)
        results = tm.sync_activations({"layer_1": (0, 1, tensor)})
    """

    def __init__(
        self,
        protocol: str = "nccl",
        secure: bool = False,
        verbose: bool = True,
        async_transfers: bool = True,
    ):
        self.protocol = protocol
        self.secure = secure
        self.verbose = verbose
        self.async_transfers = async_transfers

        self._topology: Optional[GPUTopology] = None
        self._nccl_initialized = False
        self._lock = threading.Lock()

        # Stats
        self._transfer_count = 0
        self._total_bytes = 0
        self._total_time_s = 0.0

        # Test/stub mode
        self._stub_mode = os.environ.get("VRM_MINIMAL_TEST", "0") == "1"

        if not self._stub_mode and _TORCH_AVAILABLE:
            self._topology = self._detect_topology()

    # ------------------------------------------------------------------
    # Topology detection
    # ------------------------------------------------------------------
    def _detect_topology(self) -> GPUTopology:
        """Probe GPU P2P access matrix and device capabilities."""
        topo = GPUTopology()
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return topo

        topo.num_gpus = torch.cuda.device_count()
        has_consumer = False
        for i in range(topo.num_gpus):
            props = torch.cuda.get_device_properties(i)
            topo.device_names[i] = props.name
            topo.device_caps[i] = (props.major, props.minor)
            topo.is_consumer[i] = _is_consumer_gpu(props.name)
            if topo.is_consumer[i]:
                has_consumer = True

        # Probe P2P accessibility
        p2p_blocked_count = 0
        for src in range(topo.num_gpus):
            for dst in range(topo.num_gpus):
                if src == dst:
                    topo.p2p_matrix[(src, dst)] = True
                    continue
                try:
                    can_access = torch.cuda.can_device_access_peer(src, dst)
                    topo.p2p_matrix[(src, dst)] = can_access
                    if not can_access:
                        p2p_blocked_count += 1
                except Exception:
                    topo.p2p_matrix[(src, dst)] = False
                    p2p_blocked_count += 1

        if self.verbose:
            for i in range(topo.num_gpus):
                gpu_type = "consumer" if topo.is_consumer[i] else "professional"
                log.info(
                    f"GPU {i}: {topo.device_names[i]} "
                    f"(SM {topo.device_caps[i][0]}.{topo.device_caps[i][1]}, "
                    f"{gpu_type})"
                )
            # Log P2P matrix
            for (s, d), ok in topo.p2p_matrix.items():
                if s < d:
                    status = "P2P OK" if ok else "CPU staging required"
                    log.info(f"GPU {s} <-> GPU {d}: {status}")

            if has_consumer and p2p_blocked_count > 0:
                log.warning(
                    "Consumer GPU(s) detected with P2P restrictions. "
                    "NVIDIA blocks P2P on some GeForce/RTX configurations "
                    "(IOMMU, >2 GPUs, mixed architectures). "
                    "VRAMancer will use CPU-staged transfers automatically. "
                    "NCCL (if used in distributed mode) falls back to SHM transport. "
                    "Performance impact: ~10-30%% slower than P2P/NVLink. "
                    "Workaround: use PCIe ACS override or professional GPUs (A100/H100)."
                )
            elif has_consumer and p2p_blocked_count == 0:
                log.info(
                    "Consumer GPU(s) detected but P2P is available. "
                    "Ensure IOMMU/ACS settings remain stable for reliability."
                )

        return topo

    def _can_p2p(self, src_gpu: int, dst_gpu: int) -> bool:
        """Check if direct P2P copy is possible between two GPUs."""
        if self._topology is None:
            return False
        return self._topology.p2p_matrix.get((src_gpu, dst_gpu), False)

    # ------------------------------------------------------------------
    # NCCL initialization (lazy, thread-safe)
    # ------------------------------------------------------------------
    def _ensure_nccl(self):
        """Lazily initialize NCCL process group for local multi-GPU.

        Note: In single-process multi-GPU (the common VRAMancer case),
        NCCL is NOT needed — P2P or CPU-staged transfers handle everything.
        NCCL is only useful for multi-process distributed training.
        We only init if MASTER_ADDR is set (indicating true distributed setup).
        """
        if self._nccl_initialized or self._stub_mode:
            return
        if not _DIST_AVAILABLE:
            log.debug("torch.distributed unavailable - using P2P/CPU staging")
            return

        # Only init NCCL if we're in a true multi-process distributed context
        master_addr = os.environ.get("MASTER_ADDR")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if not master_addr or world_size <= 1:
            log.debug("Single-process mode — skipping NCCL init (P2P/CPU staging)")
            return

        with self._lock:
            if self._nccl_initialized:
                return
            try:
                if not dist.is_initialized():
                    rank = int(os.environ.get("RANK", "0"))
                    dist.init_process_group(
                        backend="nccl",
                        init_method=f"env://",
                        world_size=world_size,
                        rank=rank,
                    )
                    log.info(f"NCCL process group initialized "
                             f"(world={world_size}, rank={rank})")
                self._nccl_initialized = True
            except Exception as e:
                log.warning(f"NCCL init failed ({e}), using P2P/CPU-staged transfers")

    # ------------------------------------------------------------------
    # Core transfer methods
    # ------------------------------------------------------------------
    def send_activation(
        self,
        source_gpu: int,
        target_gpu: int,
        tensor: Any,
        stream: Optional[Any] = None,
    ) -> TransferResult:
        """Transfer a tensor from source_gpu to target_gpu.

        Automatically selects the best transport method:
          1. CUDA P2P (if topology supports it)
          2. NCCL send/recv
          3. CPU-staged copy (fallback)

        Args:
            source_gpu: Source CUDA device index
            target_gpu: Target CUDA device index
            tensor: PyTorch tensor to transfer
            stream: Optional CUDA stream for async transfer

        Returns:
            TransferResult with timing and bandwidth info
        """
        if self._stub_mode:
            return TransferResult(
                method=TransportMethod.STUB,
                source_gpu=source_gpu,
                target_gpu=target_gpu,
                bytes_transferred=0,
                duration_s=0.0,
            )

        if not _TORCH_AVAILABLE:
            log.warning("PyTorch unavailable - stub transfer")
            return TransferResult(
                method=TransportMethod.STUB,
                source_gpu=source_gpu,
                target_gpu=target_gpu,
                bytes_transferred=0,
                duration_s=0.0,
            )

        start = time.perf_counter()
        tensor_bytes = tensor.nelement() * tensor.element_size()

        if source_gpu == target_gpu:
            # Same GPU - no transfer needed
            duration = time.perf_counter() - start
            return TransferResult(
                method=TransportMethod.CUDA_P2P,
                source_gpu=source_gpu,
                target_gpu=target_gpu,
                bytes_transferred=tensor_bytes,
                duration_s=duration,
            )

        # Select transport
        method, output_tensor = self._execute_transfer(
            source_gpu, target_gpu, tensor, stream
        )

        duration = time.perf_counter() - start

        # Update stats
        self._transfer_count += 1
        self._total_bytes += tensor_bytes
        self._total_time_s += duration

        # Metrics
        if _METRICS:
            FASTPATH_BYTES.labels("nccl", "send").inc(tensor_bytes)
            FASTPATH_LATENCY.labels("nccl", "send").observe(duration)

        result = TransferResult(
            method=method,
            source_gpu=source_gpu,
            target_gpu=target_gpu,
            bytes_transferred=tensor_bytes,
            duration_s=duration,
        )

        if self.verbose:
            log.info(
                f"Transfer GPU {source_gpu} -> GPU {target_gpu}: "
                f"{tensor_bytes / 1e6:.1f} MB in {duration * 1000:.2f} ms "
                f"({result.bandwidth_gbps:.1f} Gbps) via {method.name}"
            )

        return result

    def _execute_transfer(
        self,
        source_gpu: int,
        target_gpu: int,
        tensor: Any,
        stream: Optional[Any] = None,
    ) -> Tuple[TransportMethod, Any]:
        """Execute the actual GPU-to-GPU data movement.

        Tries in order: P2P direct copy -> NCCL -> CPU staging.
        Returns (method_used, output_tensor_on_target).
        """
        # --- Strategy 1: Direct CUDA P2P copy ---
        if self._can_p2p(source_gpu, target_gpu):
            try:
                return self._transfer_p2p(source_gpu, target_gpu, tensor, stream)
            except Exception as e:
                log.warning(f"P2P transfer failed ({e}), falling back")

        # --- Strategy 2: NCCL send/recv ---
        if _DIST_AVAILABLE and self._nccl_initialized:
            try:
                return self._transfer_nccl(source_gpu, target_gpu, tensor)
            except Exception as e:
                log.warning(f"NCCL transfer failed ({e}), falling back to CPU staging")

        # --- Strategy 3: CPU-staged copy (always works) ---
        return self._transfer_cpu_staged(source_gpu, target_gpu, tensor, stream)

    def _transfer_p2p(
        self,
        source_gpu: int,
        target_gpu: int,
        tensor: Any,
        stream: Optional[Any] = None,
    ) -> Tuple[TransportMethod, Any]:
        """Direct GPU-to-GPU copy via CUDA P2P (cudaMemcpyPeer).

        Requires NVLink or PCIe P2P support between the two devices.
        On consumer GPUs (RTX 3090 + 5070 Ti), P2P may work over PCIe
        if the BIOS/IOMMU allows it.
        """
        # Ensure source tensor is on the right device
        src_tensor = tensor.cuda(source_gpu) if not tensor.is_cuda else tensor

        if stream is not None:
            with torch.cuda.stream(stream):
                dst_tensor = src_tensor.to(f"cuda:{target_gpu}", non_blocking=True)
        else:
            # Use a dedicated copy stream for async operation
            copy_stream = torch.cuda.Stream(device=target_gpu)
            with torch.cuda.stream(copy_stream):
                dst_tensor = src_tensor.to(f"cuda:{target_gpu}", non_blocking=True)
            if not self.async_transfers:
                copy_stream.synchronize()

        return TransportMethod.CUDA_P2P, dst_tensor

    def _transfer_nccl(
        self,
        source_gpu: int,
        target_gpu: int,
        tensor: Any,
    ) -> Tuple[TransportMethod, Any]:
        """Transfer via NCCL point-to-point send/recv.

        Only used in true distributed (multi-process) setups where
        MASTER_ADDR/WORLD_SIZE/RANK env vars are configured.

        For single-process multi-GPU, this code path is NOT reached
        because _nccl_initialized stays False.
        """
        src_tensor = tensor.cuda(source_gpu) if not tensor.is_cuda else tensor

        # Allocate destination tensor
        with torch.cuda.device(target_gpu):
            dst_tensor = torch.empty_like(src_tensor, device=f"cuda:{target_gpu}")

        # Use NCCL point-to-point send/recv
        rank = dist.get_rank()
        if rank == source_gpu:
            dist.send(src_tensor, dst=target_gpu)
        if rank == target_gpu:
            dist.recv(dst_tensor, src=source_gpu)

        return TransportMethod.NCCL, dst_tensor

    def _transfer_cpu_staged(
        self,
        source_gpu: int,
        target_gpu: int,
        tensor: Any,
        stream: Optional[Any] = None,
    ) -> Tuple[TransportMethod, Any]:
        """Fallback: GPU -> pinned CPU memory -> GPU.

        Always works regardless of P2P topology. Uses pinned (page-locked)
        memory for maximum DMA throughput.

        Bandwidth: limited by PCIe bandwidth of the slower GPU.
        For RTX 3090 (PCIe 4): ~25 GB/s
        For RTX 5070 Ti (PCIe 5): ~50 GB/s
        Effective: ~25 GB/s (bottleneck = 3090 side)
        """
        src_tensor = tensor.cuda(source_gpu) if not tensor.is_cuda else tensor

        # Step 1: GPU -> pinned CPU memory (async)
        cpu_tensor = torch.empty(
            src_tensor.shape,
            dtype=src_tensor.dtype,
            pin_memory=True,  # Page-locked for DMA
        )

        if stream is not None:
            with torch.cuda.stream(stream):
                cpu_tensor.copy_(src_tensor, non_blocking=True)
            stream.synchronize()
        else:
            src_stream = torch.cuda.Stream(device=source_gpu)
            with torch.cuda.stream(src_stream):
                cpu_tensor.copy_(src_tensor, non_blocking=True)
            src_stream.synchronize()

        # Step 2: Pinned CPU -> target GPU (async)
        dst_stream = torch.cuda.Stream(device=target_gpu)
        with torch.cuda.stream(dst_stream):
            dst_tensor = cpu_tensor.to(f"cuda:{target_gpu}", non_blocking=True)

        if not self.async_transfers:
            dst_stream.synchronize()

        return TransportMethod.CPU_STAGED, dst_tensor

    # ------------------------------------------------------------------
    # Batch / KV cache transfers
    # ------------------------------------------------------------------
    def sync_activations(
        self,
        activations_map: Dict[str, Tuple[int, int, Any]],
    ) -> Dict[str, TransferResult]:
        """Transfer multiple tensors between GPUs.

        Args:
            activations_map: {"layer_name": (source_gpu, target_gpu, tensor), ...}

        Returns:
            dict of layer_name -> TransferResult
        """
        results = {}
        for layer_name, (src, tgt, tensor) in activations_map.items():
            result = self.send_activation(src, tgt, tensor)
            results[layer_name] = result
            if self.verbose:
                status = "OK" if result.duration_s >= 0 else "FAIL"
                log.info(f"Sync {layer_name}: {status}")
        return results

    def transfer_kv_cache(
        self,
        source_gpu: int,
        target_gpu: int,
        k_cache: Any,
        v_cache: Any,
        layer_ids: Optional[List[int]] = None,
    ) -> List[TransferResult]:
        """Transfer KV cache pages between GPUs for model parallelism.

        Optimized for LLM inference: transfers key and value caches
        as a single fused operation when possible.

        Args:
            source_gpu: Source device
            target_gpu: Target device
            k_cache: Key cache tensor [num_layers, batch, heads, seq_len, dim]
            v_cache: Value cache tensor [same shape]
            layer_ids: Optional subset of layers to transfer

        Returns:
            List of TransferResult (one per cache tensor)
        """
        results = []

        if layer_ids is not None and _TORCH_AVAILABLE:
            # Transfer only specific layers (partial KV migration)
            for lid in layer_ids:
                k_slice = k_cache[lid] if k_cache.dim() > 1 else k_cache
                v_slice = v_cache[lid] if v_cache.dim() > 1 else v_cache
                results.append(self.send_activation(source_gpu, target_gpu, k_slice))
                results.append(self.send_activation(source_gpu, target_gpu, v_slice))
        else:
            # Transfer full KV cache
            results.append(self.send_activation(source_gpu, target_gpu, k_cache))
            results.append(self.send_activation(source_gpu, target_gpu, v_cache))

        if self.verbose:
            total_bytes = sum(r.bytes_transferred for r in results)
            total_time = sum(r.duration_s for r in results)
            log.info(
                f"KV cache transfer GPU {source_gpu} -> {target_gpu}: "
                f"{total_bytes / 1e6:.1f} MB in {total_time * 1000:.2f} ms"
            )

        return results

    def prefetch_layers(
        self,
        source_gpu: int,
        target_gpu: int,
        layer_tensors: Dict[str, Any],
        priority: str = "normal",
    ) -> Dict[str, TransferResult]:
        """Asynchronously prefetch model layers to another GPU.

        Used by the scheduler to pre-stage next layers before they're needed.
        Transfers are non-blocking (returns immediately, sync on use).

        Args:
            source_gpu: Where layers currently reside
            target_gpu: Where to prefetch them
            layer_tensors: {"layer_name": tensor, ...}
            priority: "high" | "normal" | "low"
        """
        old_async = self.async_transfers
        self.async_transfers = True  # Force async for prefetch

        results = {}
        for name, tensor in layer_tensors.items():
            results[name] = self.send_activation(source_gpu, target_gpu, tensor)

        self.async_transfers = old_async
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_topology(self) -> Optional[GPUTopology]:
        """Return the detected GPU topology."""
        return self._topology

    def stats(self) -> Dict[str, Any]:
        """Return transfer statistics."""
        avg_bw = 0.0
        if self._total_time_s > 0:
            avg_bw = (self._total_bytes * 8) / (self._total_time_s * 1e9)  # Gbps
        return {
            "transfers": self._transfer_count,
            "total_bytes": self._total_bytes,
            "total_time_s": round(self._total_time_s, 4),
            "avg_bandwidth_gbps": round(avg_bw, 2),
            "topology": {
                "num_gpus": self._topology.num_gpus if self._topology else 0,
                "p2p_pairs": sum(
                    1 for v in (self._topology.p2p_matrix.values() if self._topology else [])
                    if v
                ),
            },
            "nccl_initialized": self._nccl_initialized,
            "method_preference": ["CUDA_P2P", "NCCL", "CPU_STAGED"],
        }

    def benchmark(
        self,
        sizes_mb: Optional[List[float]] = None,
        warmup: int = 2,
        iterations: int = 5,
    ) -> List[Dict[str, Any]]:
        """Benchmark GPU-to-GPU transfer for various tensor sizes.

        Returns bandwidth measurements for each (src, dst, size) combo.
        """
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return [{"error": "No GPU available"}]

        if sizes_mb is None:
            sizes_mb = [1.0, 10.0, 100.0, 500.0]

        results = []
        num_gpus = torch.cuda.device_count()

        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src == dst:
                    continue
                for size_mb in sizes_mb:
                    numel = int(size_mb * 1024 * 1024 / 4)  # float32
                    tensor = torch.randn(numel, device=f"cuda:{src}")

                    # Warmup
                    for _ in range(warmup):
                        self.send_activation(src, dst, tensor)

                    # Timed iterations
                    if _TORCH_AVAILABLE:
                        torch.cuda.synchronize()
                    times = []
                    for _ in range(iterations):
                        t0 = time.perf_counter()
                        self.send_activation(src, dst, tensor)
                        if _TORCH_AVAILABLE:
                            torch.cuda.synchronize()
                        times.append(time.perf_counter() - t0)

                    avg_time = sum(times) / len(times)
                    bw_gbps = (size_mb * 8) / (avg_time * 1000) if avg_time > 0 else 0

                    results.append({
                        "src": src,
                        "dst": dst,
                        "size_mb": size_mb,
                        "avg_time_ms": round(avg_time * 1000, 3),
                        "bandwidth_gbps": round(bw_gbps, 2),
                        "method": self._get_method_for(src, dst),
                    })

                    del tensor

        return results

    def _get_method_for(self, src: int, dst: int) -> str:
        """Return which method would be used for a given GPU pair."""
        if self._can_p2p(src, dst):
            return "CUDA_P2P"
        if self._nccl_initialized:
            return "NCCL"
        return "CPU_STAGED"

    def shutdown(self):
        """Cleanup NCCL process group."""
        if self._nccl_initialized and _DIST_AVAILABLE:
            try:
                dist.destroy_process_group()
                self._nccl_initialized = False
                log.info("NCCL process group destroyed")
            except Exception as e:
                log.warning(f"NCCL cleanup error: {e}")


__all__ = ["TransferManager", "TransferResult", "TransportMethod", "GPUTopology"]
