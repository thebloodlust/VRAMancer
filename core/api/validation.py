"""Input validation helpers for VRAMancer API endpoints."""
from __future__ import annotations

from typing import Optional, Tuple


def validate_generation_params(data: dict) -> Tuple[dict, Optional[Tuple[str, int]]]:
    """Validate and clamp generation parameters.

    Returns (params_dict, error_tuple_or_None).
    """
    max_tokens = data.get('max_tokens', data.get('max_new_tokens', 128))
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 1.0)
    top_k = data.get('top_k', 50)

    try:
        max_tokens = int(max_tokens)
        if max_tokens < 1 or max_tokens > 4096:
            return {}, ('max_tokens must be between 1 and 4096', 400)
    except (TypeError, ValueError):
        return {}, ('max_tokens must be an integer', 400)

    try:
        temperature = float(temperature)
        if temperature < 0.0 or temperature > 2.0:
            return {}, ('temperature must be between 0.0 and 2.0', 400)
    except (TypeError, ValueError):
        return {}, ('temperature must be a number', 400)

    try:
        top_p = float(top_p)
        if top_p < 0.0 or top_p > 1.0:
            return {}, ('top_p must be between 0.0 and 1.0', 400)
    except (TypeError, ValueError):
        return {}, ('top_p must be a number', 400)

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 50

    return {
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
    }, None


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens using tokenizer if available, else whitespace fallback."""
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return len(text.split())


__all__ = ["validate_generation_params", "count_tokens"]
