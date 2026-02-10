# core/stream_manager.py
"""Production layer streaming manager — prefetch, swap, and compress layers
for multi-GPU inference with adaptive memory management.

Dependencies (now implemented):
  - SimpleScheduler.allocate_block(size_mb, priority) -> AllocatedBlock
  - SimpleScheduler.release_block(block) -> None
  - SimpleScheduler.predict_next_layers(layers, lookahead) -> list[int]
  - SimpleScheduler.find_alternate_gpu(exclude) -> int
  - SimpleScheduler.migrate_block(block, target_gpu) -> None
  - GPUMonitor.vram_usage(idx) -> float (0.0-1.0)
  - GPUMonitor.detect_overload(threshold) -> Optional[int]
"""

from __future__ import annotations

import os
import time
import logging
import threading
from typing import Dict, List, Any, Optional

try:
    from core.logger import LoggerAdapter
    _logger = LoggerAdapter("stream")
except Exception:
    _logger = logging.getLogger("vramancer.stream")  # type: ignore

try:
    from core.metrics import MEMORY_PROMOTIONS, MEMORY_EVICTIONS
    _METRICS = True
except Exception:
    _METRICS = False


class StreamManager:
    """Layer streaming manager for multi-GPU inference.

    Handles:
      - Layer prefetching (predict next layers and load ahead)
      - Layer swapping (move layers between GPUs when overloaded)
      - Unused layer eviction (release memory for new layers)
      - Live monitoring with adaptive thresholds

    Usage:
        scheduler = SimpleScheduler(blocks)
        monitor = GPUMonitor()
        stream = StreamManager(scheduler=scheduler, monitor=monitor)
        stream.start_monitoring(interval=1.0)

        # During inference
        stream.preload_layer(layer_info)
        stream.prefetch_layers(current_layers, lookahead=3)
        stream.swap_if_needed()
    """

    def __init__(self, scheduler=None, monitor=None, logger=None,
                 verbose: bool = True, compressor=None,
                 overload_threshold: float = 0.90,
                 eviction_threshold: float = 0.85):
        self.scheduler = scheduler
        self.monitor = monitor
        self.logger = logger or _logger
        self.verbose = verbose
        self.compressor = compressor
        self.overload_threshold = overload_threshold
        self.eviction_threshold = eviction_threshold

        self.loaded_layers: Dict[str, Any] = {}  # layer_name -> AllocatedBlock
        self._active_layers: List[int] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._shutdown = threading.Event()

        # Stats
        self._stats = {
            "preloads": 0,
            "evictions": 0,
            "swaps": 0,
            "prefetches": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------
    # Layer lifecycle
    # ------------------------------------------------------------------

    def preload_layer(self, layer_info: Dict[str, Any]) -> bool:
        """Preload a layer onto the best available GPU.

        Parameters
        ----------
        layer_info : dict
            Must contain: "name" (str), "size_mb" (float).
            Optional: "priority" (int, default 5), "module" (nn.Module).

        Returns
        -------
        bool
            True if successfully preloaded.
        """
        if self.scheduler is None:
            self.logger.warning("No scheduler available for preload")
            return False

        name = layer_info.get("name", "unknown")
        size_mb = layer_info.get("size_mb", 100)
        priority = layer_info.get("priority", 5)
        module = layer_info.get("module")

        with self._lock:
            # Already loaded?
            if name in self.loaded_layers:
                self.logger.debug("Layer %s already loaded", name)
                return True

            try:
                block = self.scheduler.allocate_block(
                    size_mb=size_mb,
                    priority=priority,
                    layer_name=name,
                    module=module,
                )
                self.loaded_layers[name] = block
                self._stats["preloads"] += 1

                if _METRICS:
                    MEMORY_PROMOTIONS.labels("disk", "gpu").inc()

                if self.verbose:
                    self.logger.info("Preloaded %s (%.1fMB) on GPU %d",
                                     name, size_mb, block.gpu_id)
                return True
            except Exception as exc:
                self._stats["errors"] += 1
                self.logger.warning("Failed to preload %s: %s", name, exc)
                return False

    def release_layer(self, layer_name: str) -> bool:
        """Release a loaded layer and free its memory.

        Returns True if the layer was found and released.
        """
        if self.scheduler is None:
            return False

        with self._lock:
            block = self.loaded_layers.pop(layer_name, None)
            if block is None:
                return False

            try:
                self.scheduler.release_block(block)
                self._stats["evictions"] += 1
                if _METRICS:
                    MEMORY_EVICTIONS.labels("gpu", "released").inc()
                if self.verbose:
                    self.logger.info("Released layer %s from GPU %d",
                                     layer_name, block.gpu_id)
                return True
            except Exception as exc:
                self._stats["errors"] += 1
                self.logger.warning("Failed to release %s: %s", layer_name, exc)
                return False

    def prefetch_layers(self, current_layers: List[int], lookahead: int = 3) -> int:
        """Predict and preload upcoming layers.

        Parameters
        ----------
        current_layers : list of int
            Indices of currently active layers.
        lookahead : int
            How many layers ahead to prefetch.

        Returns
        -------
        int
            Number of layers successfully prefetched.
        """
        if self.scheduler is None:
            return 0

        self._active_layers = current_layers
        predicted = self.scheduler.predict_next_layers(current_layers, lookahead)
        prefetched = 0

        for layer_idx in predicted:
            name = f"layer_{layer_idx}"
            if name not in self.loaded_layers:
                # Estimate size (could be enhanced with model metadata)
                size_mb = self._estimate_layer_size(layer_idx)
                success = self.preload_layer({
                    "name": name,
                    "size_mb": size_mb,
                    "priority": 3,  # lower priority than active layers
                    "module": self._get_block_module(layer_idx),
                })
                if success:
                    prefetched += 1

        self._stats["prefetches"] += prefetched
        if self.verbose and prefetched > 0:
            self.logger.info("Prefetched %d layers (predicted: %s)", prefetched, predicted)
        return prefetched

    def unload_unused(self, active_layer_names: List[str]) -> int:
        """Release all layers not in the active set.

        Returns the number of layers unloaded.
        """
        unloaded = 0
        with self._lock:
            names_to_release = [
                name for name in list(self.loaded_layers.keys())
                if name not in active_layer_names
            ]

        for name in names_to_release:
            if self.release_layer(name):
                unloaded += 1

        if unloaded > 0:
            self.logger.info("Unloaded %d unused layers", unloaded)
        return unloaded

    # ------------------------------------------------------------------
    # Swap / migration
    # ------------------------------------------------------------------

    def swap_if_needed(self) -> bool:
        """Check for overloaded GPUs and migrate blocks if needed.

        Returns True if a swap was performed.
        """
        if self.scheduler is None or self.monitor is None:
            return False

        overloaded_gpu = self.monitor.detect_overload(self.overload_threshold)
        if overloaded_gpu is None:
            return False

        self.logger.warning("GPU %d overloaded (>%.0f%%), initiating swap",
                            overloaded_gpu, self.overload_threshold * 100)

        # Find alternate GPU
        target_gpu = self.scheduler.find_alternate_gpu(exclude=overloaded_gpu)
        if target_gpu == overloaded_gpu:
            self.logger.warning("No alternate GPU available for swap")
            return False

        # Find lowest-priority block on overloaded GPU to migrate
        with self._lock:
            candidates = [
                (name, block) for name, block in self.loaded_layers.items()
                if block.gpu_id == overloaded_gpu
            ]

        if not candidates:
            return False

        # Sort by priority (highest number = lowest priority = migrate first)
        candidates.sort(key=lambda x: -x[1].priority)
        name, block = candidates[0]

        try:
            self.scheduler.migrate_block(block, target_gpu)
            self._stats["swaps"] += 1
            self.logger.info("Swapped %s: GPU %d -> GPU %d",
                             name, overloaded_gpu, target_gpu)
            return True
        except Exception as exc:
            self.logger.error("Swap failed for %s: %s", name, exc)
            return False

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start background monitoring thread that auto-swaps when needed."""
        if self._monitoring:
            return
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,),
            daemon=True, name="stream-monitor"
        )
        self._monitor_thread.start()
        self.logger.info("Stream monitoring started (interval=%.1fs)", interval)

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        self._shutdown.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self.logger.info("Stream monitoring stopped")

    def _monitor_loop(self, interval: float) -> None:
        while self._monitoring and not self._shutdown.is_set():
            try:
                # Check for overload and auto-swap
                self.swap_if_needed()

                # Auto-evict if above eviction threshold
                if self.monitor:
                    for gpu_info in getattr(self.monitor, 'gpus', []):
                        idx = gpu_info.get('index', 0)
                        try:
                            usage = self.monitor.vram_usage(idx)
                            if usage > self.eviction_threshold:
                                self._evict_lowest_priority(idx)
                        except Exception:
                            pass  # GPU query can fail transiently
            except Exception as exc:
                self._stats["errors"] += 1
                self.logger.debug("Monitor loop error: %s", exc)
            self._shutdown.wait(timeout=interval)  # interruptible sleep

    def _evict_lowest_priority(self, gpu_id: int) -> None:
        """Evict the lowest-priority layer from a GPU."""
        with self._lock:
            candidates = [
                (name, block) for name, block in self.loaded_layers.items()
                if block.gpu_id == gpu_id
            ]

        if not candidates:
            return

        candidates.sort(key=lambda x: -x[1].priority)
        name, block = candidates[0]

        # Don't evict active layers
        if name in [f"layer_{i}" for i in self._active_layers]:
            return

        self.release_layer(name)
        self.logger.info("Auto-evicted %s from GPU %d (threshold exceeded)", name, gpu_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_layer_size(self, layer_idx: int) -> float:
        """Estimate layer size in MB from scheduler blocks."""
        if self.scheduler and layer_idx < len(self.scheduler.blocks):
            block = self.scheduler.blocks[layer_idx]
            try:
                params = sum(p.numel() * p.element_size() for p in block.parameters())
                return params / (1024 * 1024)
            except Exception:
                pass
        return 100.0  # default estimate

    def _get_block_module(self, layer_idx: int) -> Any:
        """Get the nn.Module for a layer index."""
        if self.scheduler and layer_idx < len(self.scheduler.blocks):
            return self.scheduler.blocks[layer_idx]
        return None

    @property
    def stats(self) -> Dict[str, int]:
        """Return streaming statistics."""
        return dict(self._stats)

    def __repr__(self) -> str:
        return (f"StreamManager(loaded={len(self.loaded_layers)}, "
                f"preloads={self._stats['preloads']}, "
                f"swaps={self._stats['swaps']}, "
                f"evictions={self._stats['evictions']})")

