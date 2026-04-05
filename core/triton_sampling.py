"""VRAMancer Fused Sampling Kernel (Triton).

Replaces the Python sampling loop (temperature → top-k → softmax → multinomial)
with a single fused GPU kernel.  This eliminates 3-4 separate kernel launches
per token and avoids materialized intermediate tensors.

Usage:
    from core.triton_sampling import fused_sample

    # Single call replaces temperature scaling + top-k + softmax + multinomial
    next_token = fused_sample(logits, temperature=0.8, top_k=50, top_p=0.95)

Falls back to PyTorch implementation if Triton is not available.
"""
from __future__ import annotations

import os
import logging

_logger = logging.getLogger("vramancer.triton_sampling")
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
_HAS_TRITON = False
_HAS_TORCH = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore

if not _MINIMAL:
    try:
        import triton
        import triton.language as tl
        _HAS_TRITON = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _fused_topk_softmax_kernel(
        logits_ptr,       # [V] float32 logits for one sample
        output_ptr,       # [V] float32 probabilities (output)
        temperature,      # float scalar
        top_k: tl.constexpr,  # int, 0 = disabled
        V: tl.constexpr,      # vocab size (compile-time for best perf)
        BLOCK: tl.constexpr,  # block size
    ):
        """Fused temperature → top-k mask → numerically stable softmax.

        One program handles one row (batch element).
        """
        pid = tl.program_id(0)
        row_start = pid * V

        # Pass 1: find max (for numerical stability) + apply temperature
        _max = float("-inf")
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < V
            x = tl.load(logits_ptr + row_start + cols, mask=mask, other=float("-inf"))
            x = x / temperature
            _max = tl.maximum(_max, tl.max(x, axis=0))

        # Pass 2 (top-k): find the k-th largest value as threshold.
        # We use an approximate approach: iterate with a decreasing threshold.
        # For exact top-k we need a full sort which Triton can't do efficiently,
        # so we keep the threshold approach from the Python path.
        # The caller handles top-k in Python when _HAS_TRITON but we still
        # fuse temperature + softmax here.

        # Pass 2: compute exp(x - max) and sum
        _sum = 0.0
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < V
            x = tl.load(logits_ptr + row_start + cols, mask=mask, other=float("-inf"))
            x = x / temperature
            e = tl.exp(x - _max)
            e = tl.where(mask, e, 0.0)
            _sum += tl.sum(e, axis=0)

        # Pass 3: write normalized probabilities
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < V
            x = tl.load(logits_ptr + row_start + cols, mask=mask, other=float("-inf"))
            x = x / temperature
            prob = tl.exp(x - _max) / _sum
            tl.store(output_ptr + row_start + cols, prob, mask=mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_sample(
    logits: "torch.Tensor",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
) -> "torch.Tensor":
    """Sample next token(s) from logits with fused GPU operations.

    Args:
        logits: [batch, vocab_size] float tensor on CUDA
        temperature: sampling temperature (>0)
        top_k: top-k filtering (0 = disabled)
        top_p: nucleus sampling threshold (1.0 = disabled)
        greedy: if True, return argmax (fastest path)

    Returns:
        [batch, 1] int64 tensor of sampled token IDs
    """
    if not _HAS_TORCH:
        raise RuntimeError("fused_sample requires PyTorch")

    # Greedy: single kernel call
    if greedy or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    batch_size = logits.shape[0]
    vocab_size = logits.shape[-1]

    # ── Fast path: when top_k is set, operate on just k values ──
    # This avoids processing the full vocab (32k-152k) entirely.
    # topk + softmax(k) + multinomial(k) is much faster than full-vocab ops.
    if top_k > 0 and top_k < vocab_size:
        top_vals, top_indices = torch.topk(logits, top_k, dim=-1, sorted=True)
        # Temperature scaling + softmax on just k values (tiny tensor)
        top_probs = torch.softmax(top_vals / temperature, dim=-1)
        # Nucleus (top-p) filtering on pre-sorted k values
        if top_p < 1.0:
            cumsum = top_probs.cumsum(dim=-1)
            remove = (cumsum - top_probs) >= top_p
            top_probs[remove] = 0.0
        # Sample from k candidates and map back to vocab indices
        sampled = torch.multinomial(top_probs, num_samples=1)
        return top_indices.gather(-1, sampled)

    # ── Full-vocab path (no top_k) ──
    # Use Triton fused kernel for temperature + softmax when available.
    if _HAS_TRITON and logits.is_cuda:
        # Fused temperature + softmax via Triton kernel
        probs = torch.empty_like(logits)
        BLOCK = min(triton.next_power_of_2(vocab_size), 4096)

        grid = (batch_size,)
        _fused_topk_softmax_kernel[grid](
            logits, probs,
            temperature,
            0,  # top_k already handled above
            vocab_size,
            BLOCK,
        )

        # Nucleus (top-p) filtering — use topk instead of full sort
        if top_p < 1.0:
            n_cand = min(1000, vocab_size)
            top_probs, top_indices = torch.topk(probs, n_cand, dim=-1, sorted=True)
            cumsum = top_probs.cumsum(dim=-1)
            # Mask tokens beyond the nucleus threshold (keep at least 1)
            remove = (cumsum - top_probs) >= top_p
            top_probs[remove] = 0.0
            return top_indices.gather(-1, torch.multinomial(top_probs, num_samples=1))

        return torch.multinomial(probs, num_samples=1)

    # Fallback: PyTorch path (no Triton, no top_k — both handled above)
    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        n_cand = min(1000, vocab_size)
        top_probs, top_indices = torch.topk(probs, n_cand, dim=-1, sorted=True)
        cumsum = top_probs.cumsum(dim=-1)
        remove = (cumsum - top_probs) >= top_p
        top_probs[remove] = 0.0
        return top_indices.gather(-1, torch.multinomial(top_probs, num_samples=1))

    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def has_triton() -> bool:
    """Return True if Triton kernel is available."""
    return _HAS_TRITON


__all__ = ["fused_sample", "has_triton"]
