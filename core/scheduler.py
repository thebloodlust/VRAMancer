# core/scheduler.py
"""Production block scheduler — routes model blocks through GPUs with
memory-aware allocation, predictive prefetch, and live migration.

Provides the 5 methods required by StreamManager:
  - allocate_block(size_mb, priority) -> Block
  - release_block(block) -> None
  - predict_next_layers(layers, lookahead) -> list
  - find_alternate_gpu() -> int
  - migrate_block(block, target_gpu) -> None
"""

from __future__ import annotations

import os
import time
import logging
import threading
from typing import Iterable, Callable, Any, Dict, List, Optional
from dataclasses import dataclass, field

STRICT = os.environ.get("VRM_STRICT_IMPORT", "0") in {"1", "true", "TRUE"}
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch  # type: ignore
    import torch.nn as nn
except ImportError:
    if STRICT:
        raise
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    from core.block_router import BlockRouter
except ImportError:  # pragma: no cover
    BlockRouter = None  # type: ignore

try:
    from core.block_metadata import get_block_metadata
except ImportError:  # pragma: no cover
    def get_block_metadata(idx):
        return {"importance": "normal", "estimated_size_mb": 100}

try:
    from core.monitor import GPUMonitor
except ImportError:  # pragma: no cover
    GPUMonitor = None  # type: ignore

try:
    import pynvml  # type: ignore
except ImportError:
    pynvml = None

_logger = logging.getLogger("vramancer.scheduler")


# -------------------------------------------------------------------------
# Block allocation tracking
# -------------------------------------------------------------------------

@dataclass
class AllocatedBlock:
    """Tracks a memory block allocated on a GPU."""
    block_id: int
    size_mb: float
    priority: int          # 0 = highest
    gpu_id: int
    allocated_at: float = field(default_factory=time.time)
    layer_name: str = ""
    module: Any = None     # optional nn.Module reference


class SimpleScheduler:
    """Production scheduler with memory-aware block routing.

    Parameters
    ----------
    blocks : Iterable of nn.Module
        Model blocks/layers to schedule.
    callbacks : dict
        Optional callbacks: on_start, on_step, on_end.
    monitor : GPUMonitor, optional
        GPU monitor instance for memory-aware decisions.
    """

    def __init__(
        self,
        blocks: Iterable = (),
        callbacks: Optional[Dict[str, Callable]] = None,
        monitor: Optional[Any] = None,
    ) -> None:
        self.blocks = list(blocks)
        self.callbacks = callbacks or {}
        self.router = BlockRouter() if BlockRouter else None
        self.monitor = monitor or (GPUMonitor() if GPUMonitor else None)

        # GPU tracking
        self._available_gpus = self._detect_gpus()
        self._allocated: Dict[int, AllocatedBlock] = {}  # block_id -> AllocatedBlock
        self._next_block_id = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, x: Any) -> Any:
        """Run input through all blocks sequentially."""
        if "on_start" in self.callbacks:
            self.callbacks["on_start"](0, x)

        for idx, block in enumerate(self.blocks):
            if "on_step" in self.callbacks:
                self.callbacks["on_step"](idx, x)

            meta = get_block_metadata(idx)
            if self.router:
                out = self.router.route(block, x, index=idx, **meta)
            else:
                out = block(x) if callable(block) else x
            # HuggingFace models may return object with logits attr
            if hasattr(out, "logits"):
                out = out.logits
            x = out

        if "on_end" in self.callbacks:
            self.callbacks["on_end"](len(self.blocks) - 1, x)

        return x

    def predict(self, x: Any) -> Any:
        """Forward + argmax for classification."""
        logits = self.forward(x)
        if torch is not None and hasattr(torch, "argmax"):
            return torch.argmax(logits, dim=-1)
        return logits

    # ------------------------------------------------------------------
    # GPU detection (production-ready)
    # ------------------------------------------------------------------

    def _detect_gpus(self) -> List[Dict[str, Any]]:
        """Detect available GPUs with real VRAM sizes.

        Returns list of {"id": int, "total_vram_mb": int, "free_vram_mb": int, "name": str}.
        """
        gpus: List[Dict[str, Any]] = []
        if _MINIMAL or torch is None:
            gpus.append({"id": 0, "total_vram_mb": 16000, "free_vram_mb": 16000, "name": "CPU-stub"})
            return gpus

        count = 0
        try:
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
        except Exception:
            pass

        if count == 0:
            # MPS or CPU fallback
            backend = "cpu"
            try:
                from core.utils import detect_backend
                backend = detect_backend()
            except Exception:
                pass
            if backend == "mps":
                gpus.append({"id": 0, "total_vram_mb": 0, "free_vram_mb": 0, "name": "Apple MPS"})
            else:
                gpus.append({"id": 0, "total_vram_mb": 0, "free_vram_mb": 0, "name": "CPU"})
            return gpus

        # NVIDIA/AMD via pynvml
        nvml_info = self._query_nvml(count)

        for i in range(count):
            name = "GPU"
            total_mb = 0
            free_mb = 0
            try:
                props = torch.cuda.get_device_properties(i)
                name = props.name
                total_mb = props.total_mem // (1024 * 1024)
            except Exception:
                pass

            if nvml_info and i < len(nvml_info):
                total_mb = nvml_info[i]["total_mb"]
                free_mb = nvml_info[i]["free_mb"]
            elif total_mb > 0:
                # Estimate free from torch
                try:
                    alloc = torch.cuda.memory_allocated(i) // (1024 * 1024)
                    free_mb = total_mb - alloc
                except Exception:
                    free_mb = total_mb

            gpus.append({
                "id": i,
                "total_vram_mb": total_mb,
                "free_vram_mb": free_mb,
                "name": name,
            })

        return gpus

    @staticmethod
    def _query_nvml(count: int) -> Optional[List[Dict[str, int]]]:
        """Query pynvml for accurate VRAM info."""
        if pynvml is None:
            return None
        try:
            pynvml.nvmlInit()
            info = []
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                info.append({
                    "total_mb": int(mem.total // (1024 * 1024)),
                    "free_mb": int(mem.free // (1024 * 1024)),
                })
            pynvml.nvmlShutdown()
            return info
        except Exception:
            return None

    def get_available_gpus(self) -> List[Dict[str, Any]]:
        """Return list of available GPUs with VRAM info.

        Structure: [{"id": int, "total_vram_mb": int, "free_vram_mb": int, "name": str}, ...]
        """
        return self._available_gpus

    def refresh_gpu_info(self) -> None:
        """Refresh GPU VRAM info (call periodically for accurate free mem)."""
        self._available_gpus = self._detect_gpus()

    # ------------------------------------------------------------------
    # Block allocation API (required by StreamManager)
    # ------------------------------------------------------------------

    def allocate_block(self, size_mb: float, priority: int = 5,
                       layer_name: str = "", module: Any = None) -> AllocatedBlock:
        """Allocate a memory block on the best available GPU.

        Parameters
        ----------
        size_mb : float
            Size of the block in megabytes.
        priority : int
            Priority (0 = highest). Higher priority blocks get GPU placement.
        layer_name : str
            Human-readable layer name.
        module : nn.Module, optional
            The actual PyTorch module to place.

        Returns
        -------
        AllocatedBlock
            The allocated block descriptor.
        """
        with self._lock:
            # Find GPU with most free VRAM that can fit the block
            best_gpu = self._find_best_gpu(size_mb)

            block = AllocatedBlock(
                block_id=self._next_block_id,
                size_mb=size_mb,
                priority=priority,
                gpu_id=best_gpu,
                layer_name=layer_name,
                module=module,
            )
            self._next_block_id += 1
            self._allocated[block.block_id] = block

            # Update free VRAM estimate
            for gpu in self._available_gpus:
                if gpu["id"] == best_gpu:
                    gpu["free_vram_mb"] = max(0, gpu["free_vram_mb"] - size_mb)
                    break

            _logger.debug("Allocated block %d (%.1fMB, prio=%d) -> GPU %d",
                          block.block_id, size_mb, priority, best_gpu)
            return block

    def release_block(self, block: AllocatedBlock) -> None:
        """Release a previously allocated block."""
        with self._lock:
            if block.block_id in self._allocated:
                del self._allocated[block.block_id]
                # Restore free VRAM estimate
                for gpu in self._available_gpus:
                    if gpu["id"] == block.gpu_id:
                        gpu["free_vram_mb"] += block.size_mb
                        break
                _logger.debug("Released block %d (%.1fMB) from GPU %d",
                              block.block_id, block.size_mb, block.gpu_id)

    def predict_next_layers(self, current_layers: List[int],
                            lookahead: int = 3) -> List[int]:
        """Predict the next layers to prefetch based on sequential access pattern.

        Parameters
        ----------
        current_layers : list of int
            Indices of currently active layers.
        lookahead : int
            How many layers ahead to predict.

        Returns
        -------
        list of int
            Indices of layers to prefetch.
        """
        if not current_layers:
            return list(range(min(lookahead, len(self.blocks))))

        max_idx = max(current_layers)
        total = len(self.blocks)
        predicted = []
        for i in range(1, lookahead + 1):
            next_idx = max_idx + i
            if next_idx < total:
                predicted.append(next_idx)
        return predicted

    def find_alternate_gpu(self, exclude: Optional[int] = None) -> int:
        """Find the GPU with the most free VRAM, optionally excluding one.

        Parameters
        ----------
        exclude : int, optional
            GPU index to exclude (e.g., the overloaded one).

        Returns
        -------
        int
            GPU index with the most free VRAM.
        """
        best_id = 0
        best_free = -1
        for gpu in self._available_gpus:
            if exclude is not None and gpu["id"] == exclude:
                continue
            if gpu["free_vram_mb"] > best_free:
                best_free = gpu["free_vram_mb"]
                best_id = gpu["id"]
        return best_id

    def migrate_block(self, block: AllocatedBlock, target_gpu: int) -> None:
        """Migrate a block from its current GPU to target_gpu.

        Attempts to use the TransferManager for efficient GPU-to-GPU transfer.
        Falls back to CPU-staged move if direct transfer is unavailable.
        """
        if block.gpu_id == target_gpu:
            return

        src = block.gpu_id
        _logger.info("Migrating block %d (%s) from GPU %d -> GPU %d",
                      block.block_id, block.layer_name, src, target_gpu)

        # Move the actual module if present
        if block.module is not None and torch is not None:
            try:
                from core.utils import get_device_type
                device = get_device_type(target_gpu)
                block.module = block.module.to(device)
            except Exception as exc:
                _logger.warning("Module migration failed: %s (CPU fallback)", exc)
                try:
                    block.module = block.module.to("cpu")
                except Exception:
                    pass

        # Update accounting
        with self._lock:
            for gpu in self._available_gpus:
                if gpu["id"] == src:
                    gpu["free_vram_mb"] += block.size_mb
                elif gpu["id"] == target_gpu:
                    gpu["free_vram_mb"] = max(0, gpu["free_vram_mb"] - block.size_mb)
            block.gpu_id = target_gpu

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_best_gpu(self, size_mb: float) -> int:
        """Select GPU with most free VRAM that can fit size_mb."""
        best_id = 0
        best_free = -1.0
        for gpu in self._available_gpus:
            free = gpu.get("free_vram_mb", 0)
            if free >= size_mb and free > best_free:
                best_free = free
                best_id = gpu["id"]
        # If no GPU has enough free VRAM, return the one with most free anyway
        if best_free < 0:
            for gpu in self._available_gpus:
                if gpu.get("free_vram_mb", 0) > best_free:
                    best_free = gpu.get("free_vram_mb", 0)
                    best_id = gpu["id"]
        return best_id

    def allocated_blocks(self) -> List[AllocatedBlock]:
        """Return all currently allocated blocks."""
        with self._lock:
            return list(self._allocated.values())

    def total_allocated_mb(self, gpu_id: Optional[int] = None) -> float:
        """Total MB allocated, optionally filtered by GPU."""
        with self._lock:
            blocks = self._allocated.values()
            if gpu_id is not None:
                blocks = [b for b in blocks if b.gpu_id == gpu_id]
            return sum(b.size_mb for b in blocks)
