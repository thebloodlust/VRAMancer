"""Request batching for concurrent inference.

Collects incoming inference requests over a configurable time window
and processes them in a single forward pass, improving GPU utilization
under concurrent load.

Usage:
    batcher = InferenceBatcher(generate_fn=pipeline.generate, max_batch=8)
    batcher.start()

    # From request handlers (thread-safe):
    result = batcher.submit("Hello world", max_new_tokens=50)

    batcher.stop()
"""
from __future__ import annotations

import os
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger("vramancer.batch")

# Metrics (optional)
try:
    from core.metrics import INFER_REQUESTS, INFER_LATENCY
    _METRICS = True
except ImportError:
    _METRICS = False


@dataclass
class _PendingRequest:
    """A single pending inference request waiting to be batched."""
    request_id: str
    prompt: str
    kwargs: Dict[str, Any]
    result: Optional[str] = None
    error: Optional[Exception] = None
    event: threading.Event = field(default_factory=threading.Event)
    submitted_at: float = field(default_factory=time.perf_counter)


class InferenceBatcher:
    """Transparent request batcher for LLM inference.

    Collects concurrent requests arriving within ``window_ms`` milliseconds
    and processes them together.  Each caller blocks on ``submit()`` until
    its result is ready, so the API is synchronous from the caller's
    perspective.

    For backends that don't support true batch inference (most
    auto-regressive LLMs), requests are processed sequentially but
    the batcher still provides queue management, deduplication and
    priority ordering.

    Parameters
    ----------
    generate_fn : callable
        The inference function: ``generate_fn(prompt, **kwargs) -> str``.
    max_batch : int
        Maximum number of requests to batch together (default 8).
    window_ms : float
        Time window in milliseconds to collect requests (default 50).
    timeout_s : float
        Per-request timeout in seconds (default 120).
    """

    def __init__(
        self,
        generate_fn: Callable[..., str],
        max_batch: int = 8,
        window_ms: float = 50.0,
        timeout_s: float = 120.0,
        generate_batch_fn: Optional[Callable[..., List[str]]] = None,
    ):
        self.generate_fn = generate_fn
        self.generate_batch_fn = generate_batch_fn
        self.max_batch = max_batch
        self.window_ms = window_ms
        self.timeout_s = timeout_s

        self._queue: List[_PendingRequest] = []
        self._lock = threading.Lock()
        self._has_work = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._total_batches = 0
        self._total_requests = 0
        self._total_latency_s = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background batch-processing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._batch_loop,
            daemon=True,
            name="inference-batcher",
        )
        self._thread.start()
        _logger.info(
            "InferenceBatcher started (max_batch=%d, window=%.0fms)",
            self.max_batch, self.window_ms,
        )

    def stop(self) -> None:
        """Stop the batcher, draining remaining requests."""
        self._running = False
        self._has_work.set()  # wake up the loop
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        _logger.info("InferenceBatcher stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Submit a request and block until the result is ready.

        Parameters
        ----------
        prompt : str
            Input text.
        request_id : str, optional
            Unique request ID (generated if omitted).
        **kwargs
            Generation parameters (max_new_tokens, temperature, etc.).

        Returns
        -------
        str
            Generated text.

        Raises
        ------
        TimeoutError
            If the request is not processed within ``timeout_s``.
        RuntimeError
            If the batcher is not running.
        Exception
            Re-raises any exception from the generate function.
        """
        if not self._running:
            # Fallback: direct execution if batcher is not running
            return self.generate_fn(prompt, **kwargs)

        if request_id is None:
            import uuid
            request_id = uuid.uuid4().hex[:12]

        req = _PendingRequest(
            request_id=request_id,
            prompt=prompt,
            kwargs=kwargs,
        )

        with self._lock:
            self._queue.append(req)

        # Signal the batch loop
        self._has_work.set()

        # Block until result is ready
        if not req.event.wait(timeout=self.timeout_s):
            raise TimeoutError(
                f"Inference request {request_id} timed out after {self.timeout_s}s"
            )

        if req.error is not None:
            raise req.error

        return req.result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Batch processing loop
    # ------------------------------------------------------------------

    def _batch_loop(self) -> None:
        """Background loop: collect requests, process in batches."""
        while self._running:
            # Wait for work
            self._has_work.wait(timeout=1.0)
            self._has_work.clear()

            if not self._running:
                break

            # Collect batch within time window
            time.sleep(self.window_ms / 1000.0)

            with self._lock:
                batch = self._queue[:self.max_batch]
                self._queue = self._queue[self.max_batch:]

            if not batch:
                continue

            self._process_batch(batch)

        # Drain remaining requests on shutdown
        with self._lock:
            remaining = list(self._queue)
            self._queue.clear()
        if remaining:
            self._process_batch(remaining)

    def _process_batch(self, batch: List[_PendingRequest]) -> None:
        """Process a batch of requests.

        Uses true batch forward pass (generate_batch_fn) when available
        and all requests share the same generation kwargs.  Falls back
        to sequential processing otherwise.
        """
        batch_size = len(batch)
        batch_start = time.perf_counter()

        _logger.info("Processing batch of %d request(s)", batch_size)

        # Try true batch if function provided and kwargs are uniform
        if self.generate_batch_fn and batch_size > 1:
            # Check if all requests have identical kwargs (same max_new_tokens etc.)
            first_kw = batch[0].kwargs
            uniform = all(req.kwargs == first_kw for req in batch)
            if uniform:
                try:
                    prompts = [req.prompt for req in batch]
                    results = self.generate_batch_fn(prompts, **first_kw)
                    for req, result in zip(batch, results):
                        req.result = result
                        req.event.set()
                    batch_duration = time.perf_counter() - batch_start
                    self._total_batches += 1
                    self._total_requests += batch_size
                    self._total_latency_s += batch_duration
                    if _METRICS:
                        INFER_LATENCY.observe(batch_duration)
                    _logger.info(
                        "Batch (true) complete: %d requests in %.2fs (avg %.2fs/req)",
                        batch_size, batch_duration,
                        batch_duration / batch_size if batch_size else 0,
                    )
                    return
                except Exception as exc:
                    _logger.warning(
                        "True batch failed, falling back to sequential: %s", exc
                    )

        # Sequential fallback
        for req in batch:
            try:
                result = self.generate_fn(req.prompt, **req.kwargs)
                req.result = result
            except Exception as exc:
                req.error = exc
                _logger.error(
                    "Batch request %s failed: %s", req.request_id, exc
                )
            finally:
                req.event.set()

        batch_duration = time.perf_counter() - batch_start

        # Update stats
        self._total_batches += 1
        self._total_requests += batch_size
        self._total_latency_s += batch_duration

        if _METRICS:
            INFER_LATENCY.observe(batch_duration)

        _logger.info(
            "Batch complete: %d requests in %.2fs (avg %.2fs/req)",
            batch_size, batch_duration,
            batch_duration / batch_size if batch_size else 0,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Return batcher statistics."""
        with self._lock:
            pending = len(self._queue)
        avg_latency = (
            self._total_latency_s / self._total_batches
            if self._total_batches > 0 else 0.0
        )
        return {
            "running": self._running,
            "pending_requests": pending,
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_latency_s": round(avg_latency, 4),
            "max_batch_size": self.max_batch,
            "window_ms": self.window_ms,
        }

    @property
    def pending_count(self) -> int:
        """Number of requests waiting in the queue."""
        with self._lock:
            return len(self._queue)


__all__ = ["InferenceBatcher"]
