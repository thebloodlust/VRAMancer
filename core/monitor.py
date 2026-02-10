# core/monitor.py
"""
Production GPU monitor — multi-accelerator (CUDA, ROCm, MPS, CPU).

Features:
  - Unified VRAM/RAM metrics across all accelerators
  - vram_usage() / detect_overload() for StreamManager integration
  - Continuous background polling with configurable interval
  - Prometheus gauge integration (GPU_MEMORY_USED)
  - ROCm-SMI fallback for AMD GPUs without pynvml
  - Thread-safe snapshot access
"""

from __future__ import annotations

import os
import time
import threading
import logging
from typing import Dict, List, Any, Optional

STRICT = os.environ.get("VRM_STRICT_IMPORT", "0") in {"1", "true", "TRUE"}
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch  # type: ignore
except ImportError:
    if STRICT:
        raise
    torch = None  # type: ignore

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
except ImportError:
    pynvml = None  # type: ignore

try:
    from core.utils import enumerate_devices, detect_backend
except ImportError:  # pragma: no cover
    def enumerate_devices():
        return [{"id": "cpu:0", "backend": "cpu", "index": 0, "name": "CPU", "total_memory": None}]
    def detect_backend():
        return "cpu"

try:
    from core.metrics import GPU_MEMORY_USED
except ImportError:  # pragma: no cover
    GPU_MEMORY_USED = None

_logger = logging.getLogger("vramancer.monitor")


# -------------------------------------------------------------------------
# ROCm-SMI fallback for AMD GPUs
# -------------------------------------------------------------------------

def _rocm_smi_memory(idx: int) -> Optional[Dict[str, int]]:
    """Query AMD GPU memory via rocm-smi CLI (fallback when pynvml absent)."""
    import subprocess
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            # rocm-smi JSON varies by version; try common keys
            card_key = f"card{idx}"
            if card_key in data:
                card = data[card_key]
                return {
                    "used": int(card.get("VRAM Total Used Memory (B)", 0)),
                    "total": int(card.get("VRAM Total Memory (B)", 0)),
                }
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


class GPUMonitor:
    """Production GPU monitor with background polling and Prometheus export.

    Usage:
        monitor = GPUMonitor()
        monitor.start_polling(interval=2.0)  # background thread
        print(monitor.vram_usage(0))          # 0.0 - 1.0
        print(monitor.detect_overload(0.9))   # returns gpu_index or None
        monitor.stop_polling()
    """

    def __init__(self, overload_threshold: float = 0.90) -> None:
        self.overload_threshold = overload_threshold
        self._backend = detect_backend()
        self._pynvml_ok = False
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._snapshots: Dict[Any, Dict[str, int]] = {}   # idx -> {used, reserved, total}

        # Initialize device list
        self.gpus: List[Dict[str, Any]] = []
        for d in enumerate_devices():
            self.gpus.append({
                "index": d["index"],
                "name": d["name"],
                "type": d["backend"],
                "total_memory": d.get("total_memory"),
            })

        # Try to init pynvml (NVIDIA + some ROCm builds)
        if pynvml is not None and not _MINIMAL:
            try:
                pynvml.nvmlInit()
                self._pynvml_ok = True
            except Exception:
                pass

        # Take initial snapshot
        self._refresh_all()

    # ------------------------------------------------------------------
    # Memory helpers (backward-compatible API)
    # ------------------------------------------------------------------

    def memory_allocated(self, idx: int | str = 0) -> int:
        """Bytes of VRAM (or unified memory) currently allocated."""
        snap = self._snapshots.get(idx)
        if snap:
            return snap["used"]
        return self._query_allocated(idx)

    def memory_reserved(self, idx: int | str = 0) -> int:
        """Bytes reserved by the allocator (CUDA only, 0 elsewhere)."""
        snap = self._snapshots.get(idx)
        if snap:
            return snap["reserved"]
        return self._query_reserved(idx)

    def total_memory(self, idx: int | str = 0) -> Optional[int]:
        """Total VRAM in bytes. None if unavailable."""
        snap = self._snapshots.get(idx)
        if snap and snap["total"] > 0:
            return snap["total"]
        return self._query_total(idx)

    # ------------------------------------------------------------------
    # Production API (required by StreamManager)
    # ------------------------------------------------------------------

    def vram_usage(self, idx: int | str = 0) -> float:
        """Return VRAM usage ratio 0.0 (empty) to 1.0 (full).

        Uses memory_reserved (not allocated) for accurate reporting:
        PyTorch's caching allocator reserves much more than it allocates,
        and reserved memory is what actually prevents new allocations.
        Falls back to allocated if reserved is unavailable.
        """
        total = self.total_memory(idx)
        if not total or total <= 0:
            return 0.0
        reserved = self.memory_reserved(idx)
        if reserved and reserved > 0:
            return min(reserved / total, 1.0)
        # Fallback to allocated
        used = self.memory_allocated(idx)
        return min(used / total, 1.0)

    def detect_overload(self, threshold: Optional[float] = None) -> Optional[int]:
        """Check all GPUs and return the index of the first overloaded one.

        Returns None if no GPU exceeds the threshold.
        """
        thr = threshold if threshold is not None else self.overload_threshold
        for gpu in self.gpus:
            idx = gpu["index"]
            usage = self.vram_usage(idx)
            if usage >= thr:
                return idx
        return None

    def get_free_memory(self, idx: int | str = 0) -> int:
        """Return free VRAM in bytes."""
        total = self.total_memory(idx) or 0
        used = self.memory_allocated(idx)
        return max(total - used, 0)

    def snapshot(self) -> Dict[Any, Dict[str, Any]]:
        """Return a thread-safe copy of all GPU snapshots."""
        with self._lock:
            import copy
            return copy.deepcopy(self._snapshots)

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def start_polling(self, interval: float = 2.0) -> None:
        """Start background polling thread."""
        if self._polling:
            return
        self._polling = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, args=(interval,),
            daemon=True, name="gpu-monitor-poll"
        )
        self._poll_thread.start()
        _logger.info("GPU polling started (interval=%.1fs)", interval)

    def stop_polling(self) -> None:
        """Stop background polling."""
        self._polling = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5)
        _logger.info("GPU polling stopped")

    def _poll_loop(self, interval: float) -> None:
        while self._polling:
            try:
                self._refresh_all()
                self._publish_prometheus()
            except Exception as exc:
                _logger.debug("Poll error: %s", exc)
            time.sleep(interval)

    # ------------------------------------------------------------------
    # Internal: query backends
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        """Refresh memory snapshots for all known GPUs."""
        with self._lock:
            for gpu in self.gpus:
                idx = gpu["index"]
                self._snapshots[idx] = {
                    "used": self._query_allocated(idx),
                    "reserved": self._query_reserved(idx),
                    "total": self._query_total(idx) or 0,
                }

    def _query_allocated(self, idx: int | str) -> int:
        if _MINIMAL or torch is None:
            return 0
        try:
            if isinstance(idx, int) and torch.cuda.is_available() and idx < torch.cuda.device_count():
                return torch.cuda.memory_allocated(idx)
            if (idx == "mps" or idx == 0) and hasattr(torch, "mps") and torch.mps.is_available():
                return torch.mps.current_allocated_memory()
        except Exception:
            pass
        # ROCm-SMI fallback
        if self._backend == "rocm" and isinstance(idx, int):
            info = _rocm_smi_memory(idx)
            if info:
                return info["used"]
        return 0

    def _query_reserved(self, idx: int | str) -> int:
        if _MINIMAL or torch is None:
            return 0
        try:
            if isinstance(idx, int) and torch.cuda.is_available() and idx < torch.cuda.device_count():
                return torch.cuda.memory_reserved(idx)
        except Exception:
            pass
        return 0

    def _query_total(self, idx: int | str) -> Optional[int]:
        if _MINIMAL or torch is None:
            return None
        # pynvml (most accurate for NVIDIA/ROCm)
        if self._pynvml_ok and isinstance(idx, int):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return int(mem.total)
            except Exception:
                pass
        # torch.cuda
        try:
            if isinstance(idx, int) and torch.cuda.is_available() and idx < torch.cuda.device_count():
                return torch.cuda.get_device_properties(idx).total_mem
        except Exception:
            pass
        # MPS
        try:
            if hasattr(torch, "mps") and torch.mps.is_available():
                # PyTorch 2.1+ provides recommended_max_memory on MPS
                if hasattr(torch.mps, "recommended_max_memory"):
                    return torch.mps.recommended_max_memory()
                # Fallback: macOS sysctl
                return _macos_total_memory()
        except Exception:
            pass
        # ROCm-SMI fallback
        if self._backend == "rocm" and isinstance(idx, int):
            info = _rocm_smi_memory(idx)
            if info and info["total"] > 0:
                return info["total"]
        return None

    def _publish_prometheus(self) -> None:
        """Push current memory usage to Prometheus gauges."""
        if GPU_MEMORY_USED is None:
            return
        for gpu in self.gpus:
            idx = gpu["index"]
            snap = self._snapshots.get(idx, {})
            used = snap.get("used", 0)
            try:
                GPU_MEMORY_USED.labels(gpu=str(idx)).set(used)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # System memory (CPU/RAM)
    # ------------------------------------------------------------------

    @staticmethod
    def system_memory() -> Dict[str, int]:
        """Return system RAM info in bytes."""
        if psutil is not None:
            mem = psutil.virtual_memory()
            return {"total": mem.total, "available": mem.available, "used": mem.used}
        # Fallback: /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    parts = line.split()
                    if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:"):
                        info[parts[0].rstrip(":")] = int(parts[1]) * 1024
                return {
                    "total": info.get("MemTotal", 0),
                    "available": info.get("MemAvailable", info.get("MemFree", 0)),
                    "used": info.get("MemTotal", 0) - info.get("MemAvailable", 0),
                }
        except Exception:
            return {"total": 0, "available": 0, "used": 0}

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = ["GPUMonitor:"]
        for gpu in self.gpus:
            idx = gpu["index"]
            usage_pct = self.vram_usage(idx) * 100
            total = self.total_memory(idx)
            total_str = f"{total / 1e9:.1f}GB" if total else "N/A"
            alloc = self.memory_allocated(idx)
            alloc_str = f"{alloc / 1e9:.2f}GB" if alloc else "0"
            lines.append(
                f"  [{idx}] {gpu['name']} ({gpu['type']}) "
                f"| {alloc_str}/{total_str} ({usage_pct:.1f}%)"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    def __del__(self):
        self.stop_polling()
        if self._pynvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _macos_total_memory() -> Optional[int]:
    """Get total system memory on macOS via sysctl (shared with MPS)."""
    import subprocess
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


# =========================================================================
# GPU Hot-Plug Monitor
# =========================================================================

class GPUHotPlugMonitor:
    """Detect GPU devices being added or removed at runtime.

    Polls the GPU count periodically and fires callbacks when devices
    appear or disappear.  Works with CUDA, ROCm, and MPS backends.

    Integrates with GPUMonitor to automatically refresh the device list
    and with InferencePipeline for dynamic rebalancing.

    Usage:
        hotplug = GPUHotPlugMonitor(interval=5.0)
        hotplug.on_add(lambda info: print("GPU added:", info))
        hotplug.on_remove(lambda info: print("GPU removed:", info))
        hotplug.start()
    """

    def __init__(
        self,
        interval: float = 5.0,
        gpu_monitor: Optional[GPUMonitor] = None,
    ):
        self.interval = interval
        self.gpu_monitor = gpu_monitor
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._on_add_callbacks: List[Any] = []
        self._on_remove_callbacks: List[Any] = []

        # Take initial snapshot
        self._known_gpus: Dict[int, Dict[str, Any]] = {}
        self._refresh_known()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_add(self, callback) -> None:
        """Register callback for GPU add events. Receives dict with gpu info."""
        self._on_add_callbacks.append(callback)

    def on_remove(self, callback) -> None:
        """Register callback for GPU remove events. Receives dict with gpu info."""
        self._on_remove_callbacks.append(callback)

    def start(self) -> None:
        """Start background GPU hot-plug monitoring."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="gpu-hotplug",
        )
        self._thread.start()
        _logger.info(
            "GPU hot-plug monitoring started (interval=%.1fs, known=%d GPUs)",
            self.interval, len(self._known_gpus),
        )

    def stop(self) -> None:
        """Stop hot-plug monitoring."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        _logger.info("GPU hot-plug monitoring stopped")

    @property
    def known_gpus(self) -> Dict[int, Dict[str, Any]]:
        """Return a copy of currently known GPUs."""
        with self._lock:
            return dict(self._known_gpus)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_known(self) -> None:
        """Snapshot current GPU devices."""
        devices = self._detect_gpus()
        with self._lock:
            self._known_gpus = {d["index"]: d for d in devices}

    def _detect_gpus(self) -> List[Dict[str, Any]]:
        """Detect all available GPUs."""
        if _MINIMAL:
            return []
        result = []
        # CUDA / ROCm
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        try:
                            props = torch.cuda.get_device_properties(i)
                            result.append({
                                "index": i,
                                "name": props.name,
                                "backend": "cuda",
                                "total_memory": props.total_mem,
                                "compute_capability": f"{props.major}.{props.minor}",
                            })
                        except Exception:
                            result.append({"index": i, "name": "unknown", "backend": "cuda"})
            except Exception:
                pass
            # MPS
            try:
                if hasattr(torch, "mps") and torch.mps.is_available():
                    result.append({
                        "index": 0,
                        "name": "Apple MPS",
                        "backend": "mps",
                        "total_memory": _macos_total_memory(),
                    })
            except Exception:
                pass
        # pynvml fallback (separate from torch)
        if not result and pynvml is not None:
            try:
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode()
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    result.append({
                        "index": i,
                        "name": name,
                        "backend": "cuda",
                        "total_memory": mem.total,
                    })
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return result

    def _poll_loop(self) -> None:
        """Periodically check for GPU count changes."""
        while self._running:
            try:
                current_devices = self._detect_gpus()
                current_map = {d["index"]: d for d in current_devices}

                with self._lock:
                    old_indices = set(self._known_gpus.keys())
                    new_indices = set(current_map.keys())

                    added = new_indices - old_indices
                    removed = old_indices - new_indices

                    if added or removed:
                        self._known_gpus = current_map

                # Fire callbacks outside the lock
                for idx in added:
                    info = current_map[idx]
                    _logger.info("GPU added: [%d] %s (%s)",
                                 idx, info.get("name", "?"), info.get("backend", "?"))
                    # Refresh the linked GPUMonitor
                    if self.gpu_monitor:
                        try:
                            self.gpu_monitor.gpus = []
                            for d in enumerate_devices():
                                self.gpu_monitor.gpus.append({
                                    "index": d["index"],
                                    "name": d["name"],
                                    "type": d["backend"],
                                    "total_memory": d.get("total_memory"),
                                })
                            self.gpu_monitor._refresh_all()
                        except Exception as exc:
                            _logger.debug("Monitor refresh after hotplug failed: %s", exc)
                    for cb in self._on_add_callbacks:
                        try:
                            cb(info)
                        except Exception as exc:
                            _logger.debug("on_add callback error: %s", exc)

                for idx in removed:
                    info = {"index": idx, "name": "removed"}
                    _logger.info("GPU removed: [%d]", idx)
                    if self.gpu_monitor:
                        try:
                            self.gpu_monitor.gpus = [
                                g for g in self.gpu_monitor.gpus if g["index"] != idx
                            ]
                            self.gpu_monitor._refresh_all()
                        except Exception:
                            pass
                    for cb in self._on_remove_callbacks:
                        try:
                            cb(info)
                        except Exception as exc:
                            _logger.debug("on_remove callback error: %s", exc)

            except Exception as exc:
                _logger.debug("GPU hotplug poll error: %s", exc)
            time.sleep(self.interval)
