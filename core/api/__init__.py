"""VRAMancer API sub-package.

Extracted modules:
- registry      — Thread-safe PipelineRegistry
- validation    — Input validation helpers
- circuit_breaker — Circuit-breaker pattern for inference protection
"""

from core.api.registry import PipelineRegistry
from core.api.validation import validate_generation_params, count_tokens
from core.api.circuit_breaker import CircuitBreaker, CircuitOpenError

__all__ = [
    "PipelineRegistry",
    "validate_generation_params",
    "count_tokens",
    "CircuitBreaker",
    "CircuitOpenError",
]
