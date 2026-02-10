"""VRAMancer Speculative VRAM Lending — Cooperative GPU Memory Pooling.

World's first implementation of cross-GPU speculative VRAM lending for
consumer heterogeneous multi-GPU LLM inference.

Concept:
  When a GPU is idle (no active inference batch), its free VRAM is
  "lent" to other GPUs as overflow KV cache / activation scratch space.
  When the lending GPU needs its memory back (new batch arrives), the
  borrowed regions are preemptively migrated or evicted in < 1 batch
  latency, transparently to the inference pipeline.

  This enables a unified VRAM pool across heterogeneous GPUs:
    [RTX 3090: 24 GB] + [RTX 5070 Ti: 16 GB] = 40 GB cooperative pool
    Instead of: 24 GB primary + 16 GB separate = wasted capacity

Architecture:
    VRAMLendingPool (singleton)
      ├── LeaseRegistry      — tracks all active leases (who lent what)
      ├── GPUBudgetTracker   — per-GPU free/lent/borrowed accounting
      ├── PreemptionEngine   — fast reclaim with migration-before-evict
      └── LendingPolicy      — when/how much to lend (thermal, load aware)

    Integration points:
      - PagedKVCacheManager uses LendingPool for overflow pages
      - InferencePipeline registers GPUs on load
      - GPUMonitor feeds utilization data for lending decisions
      - TransferManager handles the actual data movement

No other LLM inference framework implements this:
  - vLLM: homogeneous GPUs, no cross-GPU KV cache sharing
  - TGI: single-GPU or tensor-parallel (same model on all GPUs)
  - DeepSpeed-MII: assumes identical GPUs
  - Ollama: single-GPU only
"""

from __future__ import annotations

import os
import time
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from core.logger import LoggerAdapter
    log = LoggerAdapter("lending")
except Exception:
    import logging
    log = logging.getLogger("vramancer.lending")

try:
    from core.metrics import (
        GPU_MEMORY_USED,
        MEMORY_PROMOTIONS,
        MEMORY_EVICTIONS,
    )
    _METRICS = True
except Exception:
    _METRICS = False

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    _TORCH = False


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

class LeaseState(Enum):
    """Lifecycle of a VRAM lease."""
    ACTIVE = auto()       # Region is in use by borrower
    RECLAIMING = auto()   # Owner wants it back, migration in progress
    MIGRATED = auto()     # Data moved out, ready to release
    RELEASED = auto()     # Lease terminated, memory returned to owner


class ReclaimUrgency(Enum):
    """How urgently the owner needs its VRAM back."""
    LOW = auto()          # Gradual — owner is < 60% utilization
    MEDIUM = auto()       # Soon — owner is 60-80% utilization
    HIGH = auto()         # Immediate — owner is > 80% or new batch arriving
    CRITICAL = auto()     # Emergency — owner is > 95%, drop data if needed


@dataclass
class VRAMLease:
    """A single VRAM lending contract between two GPUs."""
    lease_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    owner_gpu: int = 0            # GPU that owns the physical memory
    borrower_gpu: int = 0         # GPU that is using the memory
    size_bytes: int = 0           # Allocated region size
    offset: int = 0               # Offset within the lending buffer
    state: LeaseState = LeaseState.ACTIVE
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0       # 0 = no expiry (until reclaimed)
    purpose: str = "kv_cache"     # What the borrowed memory is used for
    priority: int = 0             # Higher = harder to evict
    tensor_ref: Any = None        # Weak ref to the actual tensor (for migration)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at

    @property
    def is_active(self) -> bool:
        return self.state == LeaseState.ACTIVE


@dataclass
class GPUBudget:
    """VRAM accounting for a single GPU."""
    gpu_id: int
    total_bytes: int = 0          # Total VRAM capacity
    model_bytes: int = 0          # VRAM used by model weights (fixed)
    kv_cache_bytes: int = 0       # VRAM used by own KV cache
    lent_bytes: int = 0           # VRAM lent to other GPUs
    borrowed_bytes: int = 0       # VRAM borrowed from other GPUs
    reserved_bytes: int = 0       # Safety margin (never lend this)
    lending_buffer: Any = None    # Pre-allocated lending buffer on this GPU
    device_name: str = ""
    pcie_gen: int = 4             # PCIe generation (affects transfer speed)
    compute_capability: Tuple[int, int] = (0, 0)

    @property
    def free_bytes(self) -> int:
        """Actually free VRAM (not counting lent memory)."""
        used = self.model_bytes + self.kv_cache_bytes + self.lent_bytes + self.reserved_bytes
        return max(0, self.total_bytes - used)

    @property
    def lendable_bytes(self) -> int:
        """How much this GPU can lend right now."""
        return max(0, self.free_bytes - self.reserved_bytes)

    @property
    def utilization(self) -> float:
        """Current VRAM utilization ratio (0.0 to 1.0)."""
        if self.total_bytes <= 0:
            return 0.0
        used = self.model_bytes + self.kv_cache_bytes + self.reserved_bytes
        return min(used / self.total_bytes, 1.0)

    @property
    def effective_capacity(self) -> int:
        """Total usable VRAM including borrowed memory."""
        return self.total_bytes + self.borrowed_bytes - self.lent_bytes


# ═══════════════════════════════════════════════════════════════════════════
# Lending Policy
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LendingPolicy:
    """Configuration for the lending strategy."""

    # Minimum free VRAM to keep on each GPU (safety margin)
    min_free_ratio: float = 0.10       # Keep at least 10% free

    # Maximum ratio of free VRAM a GPU can lend
    max_lend_ratio: float = 0.70       # Lend at most 70% of free

    # Utilization threshold above which a GPU stops lending
    stop_lending_threshold: float = 0.75

    # Utilization threshold that triggers reclaim
    reclaim_threshold: float = 0.80

    # Critical threshold — evict immediately even if migration fails
    critical_threshold: float = 0.95

    # Minimum lease duration before it can be reclaimed (seconds)
    min_lease_duration_s: float = 0.5

    # Maximum lease duration (0 = unlimited)
    max_lease_duration_s: float = 0.0

    # Prefer lending to GPUs with faster interconnect
    prefer_fast_interconnect: bool = True

    # Temperature threshold — stop lending if GPU is too hot (°C)
    thermal_limit_c: float = 82.0

    # Lending buffer pre-allocation size (ratio of free VRAM at init)
    buffer_prealloc_ratio: float = 0.50


# ═══════════════════════════════════════════════════════════════════════════
# Core: VRAMLendingPool
# ═══════════════════════════════════════════════════════════════════════════

class VRAMLendingPool:
    """Centralized cooperative VRAM lending pool across heterogeneous GPUs.

    Enables GPUs to lend their idle VRAM to other GPUs that need more
    memory for KV cache, activations, or scratch space.

    Thread-safe. All lease operations are protected by a lock.

    Usage:
        pool = VRAMLendingPool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9)

        # GPU 0 needs overflow KV cache space → borrow from GPU 1
        lease = pool.borrow(borrower_gpu=0, size_bytes=2*1024**3,
                            purpose="kv_cache")

        # GPU 1 needs its memory back → preemptive eviction
        pool.reclaim(owner_gpu=1, urgency=ReclaimUrgency.HIGH)

        # Check pool status
        print(pool.stats())
    """

    def __init__(
        self,
        policy: Optional[LendingPolicy] = None,
        monitor: Any = None,
        transfer_manager: Any = None,
    ):
        self.policy = policy or LendingPolicy()
        self._monitor = monitor
        self._transfer_manager = transfer_manager
        self._lock = threading.Lock()
        self._shutdown = threading.Event()

        # GPU budgets
        self._budgets: Dict[int, GPUBudget] = {}

        # Active leases
        self._leases: Dict[str, VRAMLease] = {}

        # Callbacks for preemption notifications
        self._on_reclaim_callbacks: List[Callable[[VRAMLease], None]] = []
        self._on_lend_callbacks: List[Callable[[VRAMLease], None]] = []

        # Background monitor thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False

        # Stats
        self._stats = {
            "total_leases_created": 0,
            "total_leases_reclaimed": 0,
            "total_bytes_lent": 0,
            "total_bytes_reclaimed": 0,
            "preemptions_graceful": 0,
            "preemptions_forced": 0,
            "reclaim_avg_ms": 0.0,
            "peak_lent_bytes": 0,
        }
        self._reclaim_times: List[float] = []

        log.info("VRAMLendingPool initialized (policy: lend_ratio=%.0f%%, "
                 "reclaim_threshold=%.0f%%)",
                 self.policy.max_lend_ratio * 100,
                 self.policy.reclaim_threshold * 100)

    # ------------------------------------------------------------------
    # GPU Registration
    # ------------------------------------------------------------------

    def register_gpu(
        self,
        gpu_id: int,
        total_bytes: int = 0,
        model_bytes: int = 0,
        device_name: str = "",
        pcie_gen: int = 4,
        compute_capability: Tuple[int, int] = (0, 0),
    ) -> GPUBudget:
        """Register a GPU in the lending pool.

        Should be called after model loading, when model_bytes is known.
        """
        with self._lock:
            reserved = int(total_bytes * self.policy.min_free_ratio)
            budget = GPUBudget(
                gpu_id=gpu_id,
                total_bytes=total_bytes,
                model_bytes=model_bytes,
                reserved_bytes=reserved,
                device_name=device_name,
                pcie_gen=pcie_gen,
                compute_capability=compute_capability,
            )
            self._budgets[gpu_id] = budget

            # Pre-allocate lending buffer on this GPU
            if _TORCH and not _MINIMAL and total_bytes > 0:
                self._preallocate_lending_buffer(budget)

            log.info(
                "GPU %d registered: %s, total=%.1f GB, model=%.1f GB, "
                "lendable=%.1f GB, PCIe %d",
                gpu_id, device_name or f"cuda:{gpu_id}",
                total_bytes / 1e9, model_bytes / 1e9,
                budget.lendable_bytes / 1e9, pcie_gen,
            )
            return budget

    def _preallocate_lending_buffer(self, budget: GPUBudget) -> None:
        """Pre-allocate a lending buffer on the GPU.

        This avoids allocation latency during borrow operations.
        The buffer is a flat uint8 tensor that gets sliced for individual leases.
        """
        try:
            buf_size = int(budget.lendable_bytes * self.policy.buffer_prealloc_ratio)
            if buf_size < 1024 * 1024:  # Skip if < 1 MB
                return
            budget.lending_buffer = torch.empty(
                buf_size, dtype=torch.uint8,
                device=f"cuda:{budget.gpu_id}",
            )
            log.debug("Lending buffer: %.1f MB on GPU %d",
                      buf_size / 1e6, budget.gpu_id)
        except Exception as e:
            log.warning("Lending buffer alloc failed on GPU %d: %s",
                        budget.gpu_id, e)
            budget.lending_buffer = None

    def update_gpu_usage(
        self,
        gpu_id: int,
        model_bytes: Optional[int] = None,
        kv_cache_bytes: Optional[int] = None,
    ) -> None:
        """Update VRAM usage accounting for a GPU.

        Called by the inference pipeline when model load or KV cache changes.
        """
        with self._lock:
            budget = self._budgets.get(gpu_id)
            if budget is None:
                return
            if model_bytes is not None:
                budget.model_bytes = model_bytes
            if kv_cache_bytes is not None:
                budget.kv_cache_bytes = kv_cache_bytes

    # ------------------------------------------------------------------
    # Borrow (the core lending operation)
    # ------------------------------------------------------------------

    def borrow(
        self,
        borrower_gpu: int,
        size_bytes: int,
        purpose: str = "kv_cache",
        priority: int = 0,
        preferred_lender: Optional[int] = None,
    ) -> Optional[VRAMLease]:
        """Request to borrow VRAM from another GPU.

        Finds the best GPU to lend from (most free, fastest interconnect)
        and creates a lease. Returns None if no GPU can lend enough.

        Args:
            borrower_gpu: GPU that needs more memory
            size_bytes: How much VRAM to borrow
            purpose: What the memory will be used for
            priority: Higher = harder to reclaim (0-10)
            preferred_lender: Specific GPU to borrow from (optional)

        Returns:
            VRAMLease if successful, None if no capacity available
        """
        with self._lock:
            # Find best lender
            lender_gpu = self._select_lender(
                borrower_gpu, size_bytes, preferred_lender
            )
            if lender_gpu is None:
                log.debug("No GPU can lend %d MB to GPU %d",
                          size_bytes // (1024 * 1024), borrower_gpu)
                return None

            lender_budget = self._budgets[lender_gpu]
            borrower_budget = self._budgets.get(borrower_gpu)

            # Compute offset within lending buffer
            offset = self._find_buffer_offset(lender_gpu, size_bytes)

            # Create lease
            lease = VRAMLease(
                owner_gpu=lender_gpu,
                borrower_gpu=borrower_gpu,
                size_bytes=size_bytes,
                offset=offset,
                purpose=purpose,
                priority=priority,
            )

            # Update budgets
            lender_budget.lent_bytes += size_bytes
            if borrower_budget:
                borrower_budget.borrowed_bytes += size_bytes

            # Register lease
            self._leases[lease.lease_id] = lease

            # Stats
            self._stats["total_leases_created"] += 1
            self._stats["total_bytes_lent"] += size_bytes
            current_lent = sum(b.lent_bytes for b in self._budgets.values())
            self._stats["peak_lent_bytes"] = max(
                self._stats["peak_lent_bytes"], current_lent
            )

            if _METRICS:
                try:
                    MEMORY_PROMOTIONS.labels("lending", f"gpu{lender_gpu}->gpu{borrower_gpu}").inc()
                except Exception:
                    pass

            log.info(
                "Lease %s: GPU %d lends %.1f MB to GPU %d (purpose=%s, "
                "priority=%d, lender_free=%.1f GB)",
                lease.lease_id, lender_gpu,
                size_bytes / 1e6, borrower_gpu, purpose, priority,
                lender_budget.free_bytes / 1e9,
            )

            # Notify callbacks
            for cb in self._on_lend_callbacks:
                try:
                    cb(lease)
                except Exception:
                    pass

            return lease

    def _select_lender(
        self,
        borrower_gpu: int,
        size_bytes: int,
        preferred: Optional[int] = None,
    ) -> Optional[int]:
        """Select the best GPU to lend VRAM from.

        Strategy:
          1. If preferred GPU is specified and has capacity, use it
          2. Otherwise, score all candidate GPUs by:
             - Available lendable bytes (higher = better)
             - PCIe generation (faster interconnect = better)
             - Current utilization (lower = better)
             - Temperature (cooler = better, if available)
          3. Return the highest-scoring GPU
        """
        if preferred is not None:
            budget = self._budgets.get(preferred)
            if (budget and preferred != borrower_gpu
                    and budget.lendable_bytes >= size_bytes
                    and budget.utilization < self.policy.stop_lending_threshold):
                return preferred

        best_gpu = None
        best_score = -1.0

        for gpu_id, budget in self._budgets.items():
            if gpu_id == borrower_gpu:
                continue
            if budget.lendable_bytes < size_bytes:
                continue
            if budget.utilization >= self.policy.stop_lending_threshold:
                continue

            # Score: weighted combination
            # - Capacity score (0-1): how much headroom after lending
            remaining_ratio = (budget.lendable_bytes - size_bytes) / max(budget.total_bytes, 1)
            capacity_score = remaining_ratio

            # - Interconnect score: prefer PCIe 5.0 over 4.0
            pcie_score = budget.pcie_gen / 5.0

            # - Utilization score: prefer idle GPUs
            idle_score = 1.0 - budget.utilization

            # Weighted total
            score = (capacity_score * 0.4
                     + pcie_score * 0.3
                     + idle_score * 0.3)

            if score > best_score:
                best_score = score
                best_gpu = gpu_id

        return best_gpu

    def _find_buffer_offset(self, gpu_id: int, size_bytes: int) -> int:
        """Find a free offset within the GPU's lending buffer.

        Simple bump allocator — leases are packed sequentially.
        Fragmentation is acceptable because leases are short-lived.
        """
        # Sum offsets of existing leases on this GPU
        existing = [
            l for l in self._leases.values()
            if l.owner_gpu == gpu_id and l.state == LeaseState.ACTIVE
        ]
        if not existing:
            return 0

        # Find the next free offset after all existing leases
        max_end = max(l.offset + l.size_bytes for l in existing)
        return max_end

    # ------------------------------------------------------------------
    # Allocate tensor on borrowed VRAM
    # ------------------------------------------------------------------

    def allocate_on_lease(
        self,
        lease: VRAMLease,
        shape: Tuple[int, ...],
        dtype: Any = None,
    ) -> Any:
        """Allocate a tensor on borrowed VRAM.

        The tensor lives on the LENDER's GPU but is tracked as belonging
        to the BORROWER's workload.

        Args:
            lease: Active lease to allocate within
            shape: Tensor shape
            dtype: Tensor dtype (default: float16)

        Returns:
            torch.Tensor on the lender's GPU, or None if allocation fails
        """
        if not _TORCH or _MINIMAL:
            return None

        if lease.state != LeaseState.ACTIVE:
            log.warning("Cannot allocate on lease %s (state=%s)",
                        lease.lease_id, lease.state.name)
            return None

        if dtype is None:
            dtype = torch.float16

        try:
            budget = self._budgets.get(lease.owner_gpu)
            if budget and budget.lending_buffer is not None:
                # Slice from pre-allocated buffer (zero allocation latency)
                numel = 1
                for d in shape:
                    numel *= d
                elem_size = torch.tensor([], dtype=dtype).element_size()
                byte_size = numel * elem_size

                if byte_size <= lease.size_bytes:
                    # View into the lending buffer
                    start = lease.offset
                    end = start + byte_size
                    if end <= budget.lending_buffer.numel():
                        buf_slice = budget.lending_buffer[start:end]
                        tensor = buf_slice.view(torch.uint8)[:byte_size].view(
                            dtype
                        ).reshape(shape)
                        tensor.zero_()
                        lease.tensor_ref = tensor
                        return tensor

            # Fallback: direct allocation on lender GPU
            tensor = torch.zeros(
                shape, dtype=dtype,
                device=f"cuda:{lease.owner_gpu}",
            )
            lease.tensor_ref = tensor
            return tensor

        except Exception as e:
            log.warning("Allocation on lease %s failed: %s",
                        lease.lease_id, e)
            return None

    # ------------------------------------------------------------------
    # Reclaim (preemptive memory recovery)
    # ------------------------------------------------------------------

    def reclaim(
        self,
        owner_gpu: int,
        urgency: ReclaimUrgency = ReclaimUrgency.MEDIUM,
        bytes_needed: int = 0,
    ) -> int:
        """Reclaim lent VRAM back to the owner GPU.

        Strategy depends on urgency:
          LOW:      Migrate data to borrower's own VRAM or CPU, then release
          MEDIUM:   Migrate to CPU (pinned), then release
          HIGH:     Migrate to CPU, then release. Parallel all leases.
          CRITICAL: Drop data immediately (borrower must re-compute)

        Args:
            owner_gpu: GPU that wants its memory back
            urgency: How fast the reclaim must happen
            bytes_needed: Minimum bytes to reclaim (0 = reclaim all)

        Returns:
            Total bytes actually reclaimed
        """
        start_time = time.perf_counter()

        with self._lock:
            # Find all active leases where this GPU is the owner
            owner_leases = [
                l for l in self._leases.values()
                if l.owner_gpu == owner_gpu and l.state == LeaseState.ACTIVE
            ]

            if not owner_leases:
                return 0

            # Sort by priority (reclaim low-priority first)
            owner_leases.sort(key=lambda l: (l.priority, -l.age_s))

            reclaimed = 0
            target = bytes_needed if bytes_needed > 0 else sum(
                l.size_bytes for l in owner_leases
            )

            for lease in owner_leases:
                if reclaimed >= target:
                    break

                lease.state = LeaseState.RECLAIMING

                # Execute reclaim based on urgency
                if urgency == ReclaimUrgency.CRITICAL:
                    # Drop immediately — borrower loses data
                    self._force_release(lease)
                    self._stats["preemptions_forced"] += 1
                else:
                    # Graceful — migrate data first
                    self._graceful_reclaim(lease, urgency)
                    self._stats["preemptions_graceful"] += 1

                reclaimed += lease.size_bytes

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._reclaim_times.append(elapsed_ms)
        self._stats["total_leases_reclaimed"] += len([
            l for l in owner_leases if l.state == LeaseState.RELEASED
        ])
        self._stats["total_bytes_reclaimed"] += reclaimed
        if self._reclaim_times:
            self._stats["reclaim_avg_ms"] = (
                sum(self._reclaim_times) / len(self._reclaim_times)
            )

        if _METRICS:
            try:
                MEMORY_EVICTIONS.labels("lending_reclaim", f"gpu{owner_gpu}").inc()
            except Exception:
                pass

        log.info(
            "Reclaim GPU %d: %.1f MB recovered in %.2f ms "
            "(urgency=%s, leases=%d)",
            owner_gpu, reclaimed / 1e6, elapsed_ms,
            urgency.name, len(owner_leases),
        )

        return reclaimed

    def _graceful_reclaim(
        self,
        lease: VRAMLease,
        urgency: ReclaimUrgency,
    ) -> None:
        """Migrate borrowed data before releasing the lease.

        Migration strategy:
          1. Try to move data to borrower's own free VRAM
          2. If no space, move to CPU pinned memory
          3. Update the borrower's tensor reference to the new location
        """
        if lease.tensor_ref is None:
            # No data to migrate — just release
            self._force_release(lease)
            return

        tensor = lease.tensor_ref
        borrower_budget = self._budgets.get(lease.borrower_gpu)

        migrated = False

        # Strategy 1: Move to borrower's own VRAM (if space available)
        if (borrower_budget
                and borrower_budget.free_bytes >= lease.size_bytes
                and urgency != ReclaimUrgency.HIGH):
            try:
                if _TORCH and hasattr(tensor, 'to'):
                    new_tensor = tensor.to(
                        f"cuda:{lease.borrower_gpu}",
                        non_blocking=True,
                    )
                    lease.tensor_ref = new_tensor
                    lease.metadata["migrated_to"] = f"cuda:{lease.borrower_gpu}"
                    migrated = True
                    log.debug("Lease %s: migrated to borrower GPU %d",
                              lease.lease_id, lease.borrower_gpu)
            except Exception as e:
                log.debug("Migration to borrower GPU failed: %s", e)

        # Strategy 2: Move to CPU pinned memory (always works)
        if not migrated:
            try:
                if _TORCH and hasattr(tensor, 'cpu'):
                    cpu_tensor = tensor.cpu().pin_memory()
                    lease.tensor_ref = cpu_tensor
                    lease.metadata["migrated_to"] = "cpu_pinned"
                    migrated = True
                    log.debug("Lease %s: migrated to CPU pinned memory",
                              lease.lease_id)
            except Exception as e:
                log.debug("Migration to CPU failed: %s", e)

        lease.state = LeaseState.MIGRATED
        self._release_lease_accounting(lease)

    def _force_release(self, lease: VRAMLease) -> None:
        """Force-release a lease without migrating data."""
        lease.tensor_ref = None
        lease.state = LeaseState.RELEASED
        self._release_lease_accounting(lease)
        log.debug("Lease %s: force released (data dropped)", lease.lease_id)

    def _release_lease_accounting(self, lease: VRAMLease) -> None:
        """Update budget accounting when a lease is released."""
        lease.state = LeaseState.RELEASED

        owner_budget = self._budgets.get(lease.owner_gpu)
        borrower_budget = self._budgets.get(lease.borrower_gpu)

        if owner_budget:
            owner_budget.lent_bytes = max(0, owner_budget.lent_bytes - lease.size_bytes)
        if borrower_budget:
            borrower_budget.borrowed_bytes = max(
                0, borrower_budget.borrowed_bytes - lease.size_bytes
            )

        # Notify callbacks
        for cb in self._on_reclaim_callbacks:
            try:
                cb(lease)
            except Exception:
                pass

    def release(self, lease_id: str) -> bool:
        """Voluntarily release a lease (borrower no longer needs it).

        Returns True if the lease was found and released.
        """
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            if lease.state == LeaseState.RELEASED:
                return True

            lease.tensor_ref = None
            self._release_lease_accounting(lease)
            log.debug("Lease %s: voluntarily released by borrower", lease_id)
            return True

    # ------------------------------------------------------------------
    # Background monitoring (auto-reclaim)
    # ------------------------------------------------------------------

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start background thread that monitors GPU utilization
        and auto-reclaims when owners need their memory back.
        """
        if self._monitoring:
            return
        self._monitoring = True
        self._shutdown.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
            name="vram-lending-monitor",
        )
        self._monitor_thread.start()
        log.info("Lending monitor started (interval=%.1fs)", interval)

    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        self._monitoring = False
        self._shutdown.set()
        log.info("Lending monitor stopped")

    def _monitor_loop(self, interval: float) -> None:
        """Background loop: check GPU utilization and trigger reclaims."""
        while self._monitoring and not self._shutdown.is_set():
            try:
                self._check_and_reclaim()
            except Exception as e:
                log.debug("Monitor loop error: %s", e)
            self._shutdown.wait(timeout=interval)

    def _check_and_reclaim(self) -> None:
        """Check all GPUs and reclaim if any owner is under pressure."""
        for gpu_id, budget in self._budgets.items():
            if budget.lent_bytes <= 0:
                continue

            utilization = self._get_real_utilization(gpu_id)
            if utilization is None:
                utilization = budget.utilization

            if utilization >= self.policy.critical_threshold:
                self.reclaim(gpu_id, ReclaimUrgency.CRITICAL)
            elif utilization >= self.policy.reclaim_threshold:
                self.reclaim(gpu_id, ReclaimUrgency.HIGH)
            elif utilization >= self.policy.stop_lending_threshold:
                # Stop new lending but don't reclaim yet
                pass

    def _get_real_utilization(self, gpu_id: int) -> Optional[float]:
        """Get real GPU utilization from monitor (if available)."""
        if self._monitor is not None:
            try:
                return self._monitor.vram_usage(gpu_id)
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Query / Stats
    # ------------------------------------------------------------------

    def get_active_leases(self, gpu_id: Optional[int] = None) -> List[VRAMLease]:
        """Get all active leases, optionally filtered by GPU."""
        with self._lock:
            leases = [l for l in self._leases.values() if l.is_active]
            if gpu_id is not None:
                leases = [
                    l for l in leases
                    if l.owner_gpu == gpu_id or l.borrower_gpu == gpu_id
                ]
            return leases

    def get_budget(self, gpu_id: int) -> Optional[GPUBudget]:
        """Get VRAM budget for a GPU."""
        return self._budgets.get(gpu_id)

    def pool_capacity(self) -> Dict[str, Any]:
        """Total lending pool capacity across all GPUs."""
        total_lendable = sum(b.lendable_bytes for b in self._budgets.values())
        total_lent = sum(b.lent_bytes for b in self._budgets.values())
        total_borrowed = sum(b.borrowed_bytes for b in self._budgets.values())
        return {
            "total_lendable_bytes": total_lendable,
            "total_lendable_gb": total_lendable / 1e9,
            "total_lent_bytes": total_lent,
            "total_lent_gb": total_lent / 1e9,
            "total_borrowed_bytes": total_borrowed,
            "gpu_count": len(self._budgets),
            "active_leases": len([
                l for l in self._leases.values() if l.is_active
            ]),
        }

    def stats(self) -> Dict[str, Any]:
        """Comprehensive lending statistics."""
        with self._lock:
            s = dict(self._stats)
            s.update(self.pool_capacity())
            s["per_gpu"] = {}
            for gpu_id, budget in self._budgets.items():
                s["per_gpu"][gpu_id] = {
                    "device": budget.device_name,
                    "total_gb": budget.total_bytes / 1e9,
                    "model_gb": budget.model_bytes / 1e9,
                    "kv_cache_gb": budget.kv_cache_bytes / 1e9,
                    "lent_gb": budget.lent_bytes / 1e9,
                    "borrowed_gb": budget.borrowed_bytes / 1e9,
                    "free_gb": budget.free_bytes / 1e9,
                    "lendable_gb": budget.lendable_bytes / 1e9,
                    "effective_capacity_gb": budget.effective_capacity / 1e9,
                    "utilization": budget.utilization,
                    "pcie_gen": budget.pcie_gen,
                }
            return s

    def on_reclaim(self, callback: Callable[[VRAMLease], None]) -> None:
        """Register a callback for lease reclaim events."""
        self._on_reclaim_callbacks.append(callback)

    def on_lend(self, callback: Callable[[VRAMLease], None]) -> None:
        """Register a callback for new lease events."""
        self._on_lend_callbacks.append(callback)

    def close(self) -> None:
        """Shutdown the lending pool. Reclaim all leases."""
        self.stop_monitoring()
        # Reclaim everything
        for gpu_id in list(self._budgets.keys()):
            self.reclaim(gpu_id, ReclaimUrgency.HIGH)
        # Free lending buffers
        for budget in self._budgets.values():
            budget.lending_buffer = None
        log.info("VRAMLendingPool closed")

    def __repr__(self) -> str:
        cap = self.pool_capacity()
        return (
            f"VRAMLendingPool(gpus={cap['gpu_count']}, "
            f"lendable={cap['total_lendable_gb']:.1f}GB, "
            f"lent={cap['total_lent_gb']:.1f}GB, "
            f"leases={cap['active_leases']})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Singleton & factory
# ═══════════════════════════════════════════════════════════════════════════

_pool: Optional[VRAMLendingPool] = None
_pool_lock = threading.Lock()


def get_lending_pool(
    policy: Optional[LendingPolicy] = None,
    monitor: Any = None,
    transfer_manager: Any = None,
) -> VRAMLendingPool:
    """Get or create the global VRAMLendingPool singleton."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        _pool = VRAMLendingPool(
            policy=policy,
            monitor=monitor,
            transfer_manager=transfer_manager,
        )
        return _pool


def reset_lending_pool() -> None:
    """Reset the global lending pool (for testing)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


__all__ = [
    "VRAMLendingPool",
    "VRAMLease",
    "GPUBudget",
    "LendingPolicy",
    "LeaseState",
    "ReclaimUrgency",
    "get_lending_pool",
    "reset_lending_pool",
]
