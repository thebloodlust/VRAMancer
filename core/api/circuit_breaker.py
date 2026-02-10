"""Circuit-breaker pattern for VRAMancer inference.

Prevents cascading failures: after ``failure_threshold`` consecutive
errors the breaker opens and fast-fails for ``recovery_timeout`` seconds
before trying again (half-open).

Usage::

    breaker = CircuitBreaker()

    def do_inference():
        with breaker:
            return model.generate(prompt)

Or imperatively::

    if not breaker.allow_request():
        return 503, "Service unavailable"
    try:
        result = model.generate(prompt)
        breaker.record_success()
    except Exception:
        breaker.record_failure()
        raise
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing — fast-reject
    HALF_OPEN = "half_open"  # Probing — allow one request to test


class CircuitBreaker:
    """Thread-safe circuit-breaker.

    Parameters
    ----------
    failure_threshold : int
        Consecutive failures before opening.
    recovery_timeout : float
        Seconds to wait before attempting half-open.
    success_threshold : int
        Consecutive successes in half-open before closing.
    name : str
        Label for log messages.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        name: str = "inference",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition()
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        with self._lock:
            self._maybe_transition()
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                return True
            return False  # OPEN

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("CircuitBreaker[%s]: CLOSED (recovered)", self.name)
            else:
                self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._success_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("CircuitBreaker[%s]: OPEN (half-open probe failed)", self.name)
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker[%s]: OPEN after %d consecutive failures",
                    self.name, self._failure_count,
                )

    def reset(self) -> None:
        """Force reset to closed state (for tests)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0

    def status(self) -> dict:
        with self._lock:
            self._maybe_transition()
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout_s": self.recovery_timeout,
            }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        if not self.allow_request():
            raise CircuitOpenError(
                f"CircuitBreaker[{self.name}] is OPEN — "
                f"fast-failing to prevent cascade"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False  # don't suppress exceptions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_transition(self) -> None:
        """Check if OPEN breaker should transition to HALF_OPEN (call under lock)."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("CircuitBreaker[%s]: HALF_OPEN (probing)", self.name)


class CircuitOpenError(Exception):
    """Raised when executing through an OPEN circuit breaker."""
    pass


__all__ = ["CircuitBreaker", "CircuitState", "CircuitOpenError"]
