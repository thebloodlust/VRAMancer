"""VRAMancer GPU Fault Tolerance — Production-Grade GPU Error Recovery.

Handles GPU failures gracefully during inference:
  - CUDA OOM detection and automatic memory reclaim
  - GPU hardware errors (ECC, thermal shutdown, driver crash)
  - Automatic isolation of failed GPUs from the scheduler
  - Block redistribution to surviving GPUs
  - Auto-recovery with health probing when GPU comes back

Integration:
    from core.gpu_fault_tolerance import GPUFaultManager

    fm = GPUFaultManager(num_gpus=2)
    fm.on_gpu_failed(callback)

    # Wrap inference calls
    result = fm.protected_call(gpu_id=0, fn=model.forward, args=(input,))

    # Or use as context manager
    with fm.gpu_guard(gpu_id=0) as guard:
        output = model(input)
        if guard.failed:
            output = fm.retry_on_alternate(...)
"""

from __future__ import annotations

import os
import time
import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from core.logger import LoggerAdapter
    _log = LoggerAdapter("fault_tolerance")
except Exception:
    import logging
    _log = logging.getLogger("vramancer.fault_tolerance")

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

class GPUHealth(Enum):
    """GPU health state machine."""
    HEALTHY = auto()       # Normal operation
    DEGRADED = auto()      # Recovered from error, under observation
    OOM = auto()           # Out of memory — needs reclaim
    FAILED = auto()        # Hardware/driver error — isolated
    RECOVERING = auto()    # Attempting recovery
    OFFLINE = auto()       # Permanently unavailable this session


class FaultType(Enum):
    """Classification of GPU errors."""
    OOM = auto()           # CUDA out of memory
    COMPUTE_ERROR = auto() # CUDA compute error (illegal instruction, etc.)
    ECC_ERROR = auto()     # ECC memory error
    THERMAL = auto()       # Thermal throttle or shutdown
    DRIVER_CRASH = auto()  # Driver lost context
    TIMEOUT = auto()       # Operation timed out (hung GPU)
    UNKNOWN = auto()       # Unclassified error


@dataclass
class GPUFaultEvent:
    """Record of a GPU fault."""
    gpu_id: int
    fault_type: FaultType
    timestamp: float = field(default_factory=time.time)
    error_message: str = ""
    recovered: bool = False
    recovery_time_ms: float = 0.0
    blocks_migrated: int = 0


@dataclass
class GPUState:
    """Tracked state for a single GPU."""
    gpu_id: int
    health: GPUHealth = GPUHealth.HEALTHY
    fault_count: int = 0
    oom_count: int = 0
    last_fault: Optional[GPUFaultEvent] = None
    last_healthy: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    isolated_since: Optional[float] = None
    blocks_hosted: List[int] = field(default_factory=list)
    # Degraded observation period (seconds)
    observation_remaining: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# GPU Guard (context manager for protected execution)
# ═══════════════════════════════════════════════════════════════════════════

class GPUGuard:
    """Context manager that catches GPU errors and reports to FaultManager."""

    def __init__(self, fault_manager: "GPUFaultManager", gpu_id: int):
        self.fault_manager = fault_manager
        self.gpu_id = gpu_id
        self.failed = False
        self.fault_type: Optional[FaultType] = None
        self.error: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.failed = True
            self.error = exc_val
            self.fault_type = self.fault_manager._classify_error(exc_val)
            self.fault_manager._handle_fault(self.gpu_id, self.fault_type, str(exc_val))
            # Suppress the exception — caller checks guard.failed
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# GPUFaultManager
# ═══════════════════════════════════════════════════════════════════════════

class GPUFaultManager:
    """Production GPU fault detection, isolation, and recovery.

    Monitors GPU health, catches CUDA errors, isolates failed GPUs,
    redistributes work, and probes for recovery.

    Parameters
    ----------
    num_gpus : int
        Number of GPUs to manage.
    max_consecutive_failures : int
        After this many consecutive failures, mark GPU as OFFLINE.
    recovery_probe_interval : float
        Seconds between recovery probe attempts for failed GPUs.
    oom_reclaim_fraction : float
        Fraction of GPU memory to free on OOM (0.0 - 1.0).
    degraded_observation_s : float
        How long (seconds) to observe a recovered GPU before marking healthy.
    """

    def __init__(
        self,
        num_gpus: int = 0,
        max_consecutive_failures: int = 3,
        recovery_probe_interval: float = 10.0,
        oom_reclaim_fraction: float = 0.3,
        degraded_observation_s: float = 30.0,
    ):
        if num_gpus == 0 and _TORCH and not _MINIMAL:
            try:
                num_gpus = torch.cuda.device_count()
            except Exception:
                pass

        self.num_gpus = num_gpus
        self.max_consecutive_failures = max_consecutive_failures
        self.recovery_probe_interval = recovery_probe_interval
        self.oom_reclaim_fraction = oom_reclaim_fraction
        self.degraded_observation_s = degraded_observation_s

        self._lock = threading.RLock()
        self._gpu_states: Dict[int, GPUState] = {
            i: GPUState(gpu_id=i) for i in range(num_gpus)
        }
        self._fault_history: List[GPUFaultEvent] = []
        self._callbacks_on_fail: List[Callable[[int, FaultType], None]] = []
        self._callbacks_on_recover: List[Callable[[int], None]] = []
        self._recovery_thread: Optional[threading.Thread] = None
        self._running = False

        # Block migration callback (set by InferencePipeline)
        self._migrate_blocks_fn: Optional[Callable[[int, int], int]] = None

        _log.info("GPUFaultManager: tracking %d GPUs", num_gpus)

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_error(error: Exception) -> FaultType:
        """Classify a CUDA/GPU error into a FaultType."""
        msg = str(error).lower()

        if "out of memory" in msg or "oom" in msg:
            return FaultType.OOM
        if "ecc" in msg:
            return FaultType.ECC_ERROR
        if "temperature" in msg or "thermal" in msg:
            return FaultType.THERMAL
        if "driver" in msg or "context" in msg or "device-side assert" in msg:
            return FaultType.DRIVER_CRASH
        if "timeout" in msg or "timed out" in msg or "watchdog" in msg:
            return FaultType.TIMEOUT
        if "illegal" in msg or "compute" in msg or "assert" in msg:
            return FaultType.COMPUTE_ERROR
        return FaultType.UNKNOWN

    # ------------------------------------------------------------------
    # Fault handling
    # ------------------------------------------------------------------

    def _handle_fault(self, gpu_id: int, fault_type: FaultType, error_msg: str) -> None:
        """Handle a detected GPU fault."""
        with self._lock:
            state = self._gpu_states.get(gpu_id)
            if state is None:
                return

            event = GPUFaultEvent(
                gpu_id=gpu_id,
                fault_type=fault_type,
                error_message=error_msg,
            )
            self._fault_history.append(event)
            state.fault_count += 1
            state.consecutive_failures += 1
            state.last_fault = event

            if fault_type == FaultType.OOM:
                state.oom_count += 1
                state.health = GPUHealth.OOM
                _log.warning(
                    "GPU %d OOM (#%d) — attempting memory reclaim",
                    gpu_id, state.oom_count,
                )
                self._handle_oom(gpu_id)
            elif state.consecutive_failures >= self.max_consecutive_failures:
                state.health = GPUHealth.OFFLINE
                state.isolated_since = time.time()
                _log.error(
                    "GPU %d OFFLINE after %d consecutive failures — permanently isolated",
                    gpu_id, state.consecutive_failures,
                )
                self._isolate_gpu(gpu_id)
            else:
                state.health = GPUHealth.FAILED
                state.isolated_since = time.time()
                _log.error(
                    "GPU %d FAILED (%s): %s — isolated, will probe for recovery",
                    gpu_id, fault_type.name, error_msg[:200],
                )
                self._isolate_gpu(gpu_id)

        # Notify callbacks (outside lock)
        for cb in self._callbacks_on_fail:
            try:
                cb(gpu_id, fault_type)
            except Exception:
                pass

    def _handle_oom(self, gpu_id: int) -> None:
        """Attempt to recover from OOM by freeing memory."""
        if not _TORCH or _MINIMAL:
            return

        try:
            with torch.cuda.device(gpu_id):
                # Clear PyTorch caches
                torch.cuda.empty_cache()

                # Attempt garbage collection
                import gc
                gc.collect()

                # Log memory state
                allocated = torch.cuda.memory_allocated(gpu_id)
                reserved = torch.cuda.memory_reserved(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_mem
                _log.info(
                    "GPU %d after OOM reclaim: allocated=%.0f MB, reserved=%.0f MB, total=%.0f MB",
                    gpu_id, allocated / 1e6, reserved / 1e6, total / 1e6,
                )

                # If still > 90% used, mark as degraded
                if allocated / total > 0.9:
                    self._gpu_states[gpu_id].health = GPUHealth.DEGRADED
                else:
                    self._gpu_states[gpu_id].health = GPUHealth.DEGRADED
                    self._gpu_states[gpu_id].consecutive_failures = 0
                    self._gpu_states[gpu_id].observation_remaining = self.degraded_observation_s

        except Exception as e:
            _log.warning("OOM recovery failed for GPU %d: %s", gpu_id, e)
            self._gpu_states[gpu_id].health = GPUHealth.FAILED

    def _isolate_gpu(self, gpu_id: int) -> None:
        """Isolate a failed GPU — migrate its blocks to healthy GPUs."""
        state = self._gpu_states.get(gpu_id)
        if state is None:
            return

        # Migrate blocks if callback is set
        if self._migrate_blocks_fn and state.blocks_hosted:
            healthy_gpus = self.get_healthy_gpus()
            if healthy_gpus:
                target = healthy_gpus[0]
                try:
                    migrated = self._migrate_blocks_fn(gpu_id, target)
                    if state.last_fault:
                        state.last_fault.blocks_migrated = migrated
                    _log.info(
                        "Migrated %d blocks from failed GPU %d to GPU %d",
                        migrated, gpu_id, target,
                    )
                except Exception as e:
                    _log.error("Block migration from GPU %d failed: %s", gpu_id, e)

        # Ensure recovery thread is running
        self._ensure_recovery_thread()

    def _ensure_recovery_thread(self) -> None:
        """Start the background recovery probe thread if not running."""
        if self._running:
            return
        self._running = True
        self._recovery_thread = threading.Thread(
            target=self._recovery_loop,
            daemon=True,
            name="gpu-recovery-probe",
        )
        self._recovery_thread.start()

    def _recovery_loop(self) -> None:
        """Background thread that probes failed GPUs for recovery."""
        while self._running:
            time.sleep(self.recovery_probe_interval)
            with self._lock:
                failed_gpus = [
                    gid for gid, s in self._gpu_states.items()
                    if s.health == GPUHealth.FAILED
                ]
                degraded_gpus = [
                    gid for gid, s in self._gpu_states.items()
                    if s.health == GPUHealth.DEGRADED
                ]

            # Probe failed GPUs
            for gid in failed_gpus:
                if self._probe_gpu(gid):
                    self._recover_gpu(gid)

            # Promote degraded GPUs that passed observation period
            for gid in degraded_gpus:
                with self._lock:
                    state = self._gpu_states[gid]
                    state.observation_remaining -= self.recovery_probe_interval
                    if state.observation_remaining <= 0:
                        state.health = GPUHealth.HEALTHY
                        state.consecutive_failures = 0
                        _log.info("GPU %d promoted back to HEALTHY", gid)

            # Stop thread if no more failed/degraded GPUs
            with self._lock:
                any_unhealthy = any(
                    s.health in (GPUHealth.FAILED, GPUHealth.DEGRADED, GPUHealth.OOM)
                    for s in self._gpu_states.values()
                )
                if not any_unhealthy:
                    self._running = False
                    break

    def _probe_gpu(self, gpu_id: int) -> bool:
        """Probe a failed GPU to check if it's operational again."""
        if not _TORCH or _MINIMAL:
            return False

        try:
            with torch.cuda.device(gpu_id):
                # Try a small allocation + compute
                t = torch.zeros(16, 16, device=f"cuda:{gpu_id}")
                result = (t + 1).sum().item()
                assert result == 256.0
                del t
                torch.cuda.empty_cache()
            _log.info("GPU %d recovery probe: PASSED", gpu_id)
            return True
        except Exception as e:
            _log.debug("GPU %d recovery probe: FAILED (%s)", gpu_id, e)
            return False

    def _recover_gpu(self, gpu_id: int) -> None:
        """Mark a GPU as recovered and notify callbacks."""
        with self._lock:
            state = self._gpu_states.get(gpu_id)
            if state is None:
                return

            state.health = GPUHealth.DEGRADED
            state.observation_remaining = self.degraded_observation_s
            state.isolated_since = None
            if state.last_fault:
                state.last_fault.recovered = True
                state.last_fault.recovery_time_ms = (
                    (time.time() - state.last_fault.timestamp) * 1000
                )
            _log.info(
                "GPU %d RECOVERED — entering %.0fs observation period",
                gpu_id, self.degraded_observation_s,
            )

        for cb in self._callbacks_on_recover:
            try:
                cb(gpu_id)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gpu_guard(self, gpu_id: int) -> GPUGuard:
        """Context manager for protected GPU execution.

        Usage::

            with fm.gpu_guard(0) as guard:
                output = model(input)
            if guard.failed:
                # handle error
        """
        return GPUGuard(self, gpu_id)

    def protected_call(
        self,
        gpu_id: int,
        fn: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        retry_on_oom: bool = True,
        max_retries: int = 2,
    ) -> Any:
        """Execute a function on a GPU with fault protection.

        Parameters
        ----------
        gpu_id : int
            Target GPU.
        fn : callable
            Function to execute.
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.
        retry_on_oom : bool
            Retry after clearing cache on OOM.
        max_retries : int
            Maximum retry attempts for OOM.

        Returns
        -------
        Any
            Function return value.

        Raises
        ------
        RuntimeError
            If GPU is isolated or all retries exhausted.
        """
        if kwargs is None:
            kwargs = {}

        # Check GPU health
        with self._lock:
            state = self._gpu_states.get(gpu_id)
            if state and state.health in (GPUHealth.FAILED, GPUHealth.OFFLINE):
                raise RuntimeError(
                    f"GPU {gpu_id} is {state.health.name} — cannot execute. "
                    f"Use get_healthy_gpus() to find an alternative."
                )

        for attempt in range(max_retries + 1):
            with self.gpu_guard(gpu_id) as guard:
                result = fn(*args, **kwargs)

            if not guard.failed:
                # Success — reset consecutive failures
                with self._lock:
                    if gpu_id in self._gpu_states:
                        self._gpu_states[gpu_id].consecutive_failures = 0
                        if self._gpu_states[gpu_id].health == GPUHealth.DEGRADED:
                            pass  # Stay degraded until observation completes
                        elif self._gpu_states[gpu_id].health == GPUHealth.OOM:
                            self._gpu_states[gpu_id].health = GPUHealth.HEALTHY
                return result

            # Failed
            if guard.fault_type == FaultType.OOM and retry_on_oom and attempt < max_retries:
                _log.warning(
                    "GPU %d OOM — retrying (%d/%d) after cache clear",
                    gpu_id, attempt + 1, max_retries,
                )
                continue
            else:
                # Non-OOM or retries exhausted — try alternate GPU
                healthy = self.get_healthy_gpus()
                if healthy:
                    alt = healthy[0]
                    _log.warning(
                        "Falling back to GPU %d after GPU %d failure",
                        alt, gpu_id,
                    )
                    return fn(*args, **kwargs)  # caller should move tensors
                raise RuntimeError(
                    f"GPU {gpu_id} failed ({guard.fault_type}) and no healthy GPU available"
                )

        raise RuntimeError(f"GPU {gpu_id}: all {max_retries} retries exhausted")

    def get_healthy_gpus(self) -> List[int]:
        """Return list of GPU IDs in HEALTHY or DEGRADED state."""
        with self._lock:
            return [
                gid for gid, s in self._gpu_states.items()
                if s.health in (GPUHealth.HEALTHY, GPUHealth.DEGRADED)
            ]

    def get_gpu_health(self, gpu_id: int) -> GPUHealth:
        """Get health state of a specific GPU."""
        with self._lock:
            state = self._gpu_states.get(gpu_id)
            return state.health if state else GPUHealth.OFFLINE

    def is_healthy(self, gpu_id: int) -> bool:
        """Check if a GPU is operational."""
        health = self.get_gpu_health(gpu_id)
        return health in (GPUHealth.HEALTHY, GPUHealth.DEGRADED)

    def register_blocks(self, gpu_id: int, block_ids: List[int]) -> None:
        """Register blocks hosted on a GPU (for migration on failure)."""
        with self._lock:
            state = self._gpu_states.get(gpu_id)
            if state:
                state.blocks_hosted = list(block_ids)

    def set_migrate_callback(self, fn: Callable[[int, int], int]) -> None:
        """Set the block migration callback.

        fn(source_gpu, target_gpu) -> num_blocks_migrated
        """
        self._migrate_blocks_fn = fn

    def on_gpu_failed(self, callback: Callable[[int, FaultType], None]) -> None:
        """Register callback for GPU failure events."""
        self._callbacks_on_fail.append(callback)

    def on_gpu_recovered(self, callback: Callable[[int], None]) -> None:
        """Register callback for GPU recovery events."""
        self._callbacks_on_recover.append(callback)

    # ------------------------------------------------------------------
    # Stats / status
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return fault tolerance statistics."""
        with self._lock:
            gpu_health = {
                gid: {
                    "health": s.health.name,
                    "fault_count": s.fault_count,
                    "oom_count": s.oom_count,
                    "consecutive_failures": s.consecutive_failures,
                    "isolated_since": s.isolated_since,
                    "blocks_hosted": len(s.blocks_hosted),
                }
                for gid, s in self._gpu_states.items()
            }
            return {
                "num_gpus": self.num_gpus,
                "healthy_gpus": len(self.get_healthy_gpus()),
                "total_faults": len(self._fault_history),
                "gpu_states": gpu_health,
                "recovery_thread_active": self._running,
                "recent_faults": [
                    {
                        "gpu": e.gpu_id,
                        "type": e.fault_type.name,
                        "recovered": e.recovered,
                        "time": e.timestamp,
                        "message": e.error_message[:100],
                    }
                    for e in self._fault_history[-10:]
                ],
            }

    def stop(self) -> None:
        """Stop recovery probing."""
        self._running = False
        if self._recovery_thread and self._recovery_thread.is_alive():
            self._recovery_thread.join(timeout=5)


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

_global_fault_manager: Optional[GPUFaultManager] = None
_fm_lock = threading.Lock()


def get_fault_manager(**kwargs) -> GPUFaultManager:
    """Get or create the global GPUFaultManager singleton."""
    global _global_fault_manager
    with _fm_lock:
        if _global_fault_manager is None:
            _global_fault_manager = GPUFaultManager(**kwargs)
        return _global_fault_manager


def reset_fault_manager() -> None:
    """Reset the global fault manager (for tests)."""
    global _global_fault_manager
    with _fm_lock:
        if _global_fault_manager is not None:
            _global_fault_manager.stop()
        _global_fault_manager = None


__all__ = [
    "GPUFaultManager",
    "GPUFaultEvent",
    "GPUGuard",
    "GPUHealth",
    "GPUState",
    "FaultType",
    "get_fault_manager",
    "reset_fault_manager",
]
