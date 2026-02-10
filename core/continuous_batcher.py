"""VRAMancer Continuous Batching Engine.

Implements iteration-level scheduling inspired by Orca/vLLM:
  - Multiple requests in-flight simultaneously
  - New requests join the batch at any iteration
  - Completed requests leave without blocking others
  - KV cache managed per-request via PagedAttention

This is the **#1 missing feature** that vLLM/TGI have and VRAMancer lacked.

Architecture:
    ContinuousBatcher
      ├── RequestQueue           (thread-safe intake)
      ├── ActiveBatch            (currently generating)
      ├── PagedKVCacheManager    (see paged_attention.py)
      └── IterationScheduler     (decides who runs each step)

Usage:
    batcher = ContinuousBatcher(model, tokenizer)
    batcher.start()

    # Submit requests (non-blocking)
    fut1 = batcher.submit("Hello, world!", max_new_tokens=50)
    fut2 = batcher.submit("Once upon a time", max_new_tokens=100)

    # Collect results
    print(fut1.result())  # blocks until done
    print(fut2.result())

    batcher.stop()
"""

from __future__ import annotations

import os
import time
import uuid
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import Future

_logger = logging.getLogger("vramancer.continuous_batcher")
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    _TORCH = False


# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------

class RequestStatus(Enum):
    WAITING = auto()
    ACTIVE = auto()
    FINISHED = auto()
    CANCELLED = auto()
    ERROR = auto()


@dataclass
class InferenceRequest:
    """A single generation request tracked by the batcher."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    stop_token_id: Optional[int] = None

    # Internal state
    input_ids: Any = None          # torch.Tensor [1, seq_len]
    generated_ids: Any = None      # torch.Tensor [1, gen_len] (growing)
    tokens_generated: int = 0
    kv_cache: Any = None           # per-request past_key_values
    status: RequestStatus = RequestStatus.WAITING
    future: Optional[Future] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    # Streaming callback (optional)
    on_token: Optional[Callable[[str], None]] = None


# ---------------------------------------------------------------------------
# Continuous Batcher
# ---------------------------------------------------------------------------

class ContinuousBatcher:
    """Iteration-level continuous batching scheduler.

    Each iteration:
      1. Admit new requests from the waiting queue (up to max_batch_size)
      2. Build a padded batch from all active requests
      3. Run a single forward pass for ALL active requests
      4. Scatter results back, check for completion
      5. Evict completed requests, free their KV cache slots
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        max_batch_size: int = 32,
        max_waiting_queue: int = 256,
        device: str = "auto",
        verbose: bool = True,
        paged_kv_manager: Any = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_waiting_queue = max_waiting_queue
        self.verbose = verbose
        self.paged_kv = paged_kv_manager

        # Device
        if device == "auto":
            if _TORCH and torch.cuda.is_available():
                self._device = "cuda:0"
            else:
                self._device = "cpu"
        else:
            self._device = device

        # Queues & state
        self._waiting: List[InferenceRequest] = []
        self._active: List[InferenceRequest] = []
        self._completed: List[InferenceRequest] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Metrics
        self._total_requests = 0
        self._total_tokens_generated = 0
        self._total_iterations = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Future:
        """Submit a generation request (non-blocking).

        Returns a Future whose .result() will be the generated text.
        """
        fut: Future = Future()
        req = InferenceRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            future=fut,
            on_token=on_token,
        )

        # Set stop token
        if self.tokenizer:
            eos = getattr(self.tokenizer, 'eos_token_id', None)
            req.stop_token_id = eos

        with self._lock:
            if len(self._waiting) >= self.max_waiting_queue:
                fut.set_exception(RuntimeError("Waiting queue full"))
                return fut
            self._waiting.append(req)
            self._total_requests += 1

        _logger.debug("Request %s submitted (%d waiting)", req.request_id, len(self._waiting))
        return fut

    def start(self) -> None:
        """Start the continuous batching loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="continuous-batcher",
        )
        self._thread.start()
        _logger.info("ContinuousBatcher started (max_batch=%d)", self.max_batch_size)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the batcher gracefully."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        # Cancel remaining waiting requests
        with self._lock:
            for req in self._waiting:
                if req.future and not req.future.done():
                    req.future.cancel()
                req.status = RequestStatus.CANCELLED
            self._waiting.clear()
        _logger.info("ContinuousBatcher stopped")

    def stats(self) -> Dict[str, Any]:
        """Return batcher statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "waiting": len(self._waiting),
            "active": len(self._active),
            "completed": len(self._completed),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens_generated,
            "total_iterations": self._total_iterations,
            "throughput_tok_s": (
                self._total_tokens_generated / elapsed if elapsed > 0 else 0
            ),
            "avg_batch_size": (
                self._total_tokens_generated / max(self._total_iterations, 1)
            ),
        }

    @property
    def pending_count(self) -> int:
        return len(self._waiting) + len(self._active)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main continuous batching loop — one iteration = one forward pass."""
        while self._running:
            with self._lock:
                # 1. Admit new requests
                self._admit_requests()

                if not self._active:
                    # Nothing to do — sleep briefly
                    pass
                else:
                    # 2. Run one iteration
                    try:
                        self._iteration_step()
                    except Exception as e:
                        _logger.error("Iteration error: %s", e)
                        # Mark all active as error
                        for req in self._active:
                            req.status = RequestStatus.ERROR
                            if req.future and not req.future.done():
                                req.future.set_exception(e)
                        self._active.clear()

                    # 3. Evict completed
                    self._evict_completed()

            # Yield CPU — adaptive sleep based on load
            if not self._active and not self._waiting:
                time.sleep(0.01)  # idle
            else:
                time.sleep(0.0001)  # busy

    def _admit_requests(self) -> None:
        """Move requests from waiting → active (up to batch capacity)."""
        slots = self.max_batch_size - len(self._active)
        if slots <= 0 or not self._waiting:
            return

        to_admit = self._waiting[:slots]
        self._waiting = self._waiting[slots:]

        for req in to_admit:
            try:
                self._prepare_request(req)
                req.status = RequestStatus.ACTIVE
                self._active.append(req)
                _logger.debug("Request %s admitted (batch=%d)",
                              req.request_id, len(self._active))
            except Exception as e:
                req.status = RequestStatus.ERROR
                if req.future and not req.future.done():
                    req.future.set_exception(e)

    def _prepare_request(self, req: InferenceRequest) -> None:
        """Tokenize prompt and prepare initial state."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")

        tokens = self.tokenizer(req.prompt, return_tensors="pt")
        req.input_ids = tokens["input_ids"]
        req.generated_ids = req.input_ids.clone()

        if _TORCH and self._device != "cpu":
            try:
                req.input_ids = req.input_ids.to(self._device)
                req.generated_ids = req.generated_ids.to(self._device)
            except Exception:
                pass

        # Paged attention: allocate pages + try prefix cache
        if self.paged_kv:
            try:
                token_list = req.input_ids[0].tolist()
                hits = self.paged_kv.try_prefix_cache(req.request_id, token_list)
                if hits > 0:
                    # Prefix was cached — reconstruct KV for the cached portion
                    kv = self.paged_kv.to_hf_cache(req.request_id)
                    if kv is not None:
                        req.kv_cache = kv
                        _logger.debug("Prefix cache hit: %d tokens for %s",
                                      hits, req.request_id)
                else:
                    self.paged_kv.allocate(req.request_id)
            except Exception as e:
                _logger.debug("Paged KV setup failed for %s: %s",
                              req.request_id, e)

    def _iteration_step(self) -> None:
        """Run one forward pass for all active requests.

        This is the core continuous batching logic:
        - Prefill phase: process new requests individually (variable-length)
        - Decode phase: coalesce ALL decode requests into a single padded
          forward pass for maximum GPU utilization (the key optimization
          that makes continuous batching work).
        """
        self._total_iterations += 1

        if not _TORCH or self.model is None:
            # Stub mode: advance each request by one "token"
            for req in self._active:
                req.tokens_generated += 1
                if req.tokens_generated >= req.max_new_tokens:
                    self._finish_request(req, req.prompt + " [stub output]")
            return

        # Separate prefill (no KV cache yet) from decode (have KV cache)
        prefill = [r for r in self._active if r.kv_cache is None]
        decode = [r for r in self._active if r.kv_cache is not None]

        # --- Prefill: run individually (each has different seq length) ---
        for req in prefill:
            self._forward_single(req, is_prefill=True)

        # --- Decode: coalesce into ONE batched forward pass ---
        if len(decode) >= 2:
            self._forward_batched_decode(decode)
        elif len(decode) == 1:
            self._forward_single(decode[0], is_prefill=False)

    def _forward_batched_decode(self, requests: List[InferenceRequest]) -> None:
        """Coalesce multiple decode requests into a single padded forward pass.

        This is what makes continuous batching actually faster than sequential:
        all active decode-phase requests share a single model invocation.

        Strategy:
          1. Collect last-token input_ids for each request → [batch, 1]
          2. Pad KV caches to same sequence length
          3. Build attention_mask accounting for padding
          4. Single model forward
          5. Scatter logits back to individual requests
        """
        try:
            batch_size = len(requests)

            # Step 1: Collect inputs (each is [1, 1] — the last token)
            inputs = [req.generated_ids[:, -1:] for req in requests]
            batched_input = torch.cat(inputs, dim=0)  # [batch, 1]

            # Step 2: Check if we can batch KV caches
            # HuggingFace KV cache: tuple of (num_layers x (key, value))
            # key/value shape: [batch, num_heads, seq_len, head_dim]
            # For batched decode, we need matching seq_len or we pad.
            kv_shapes = []
            for req in requests:
                if req.kv_cache and isinstance(req.kv_cache, tuple) and len(req.kv_cache) > 0:
                    # First layer, first element (key or value)
                    first_kv = req.kv_cache[0]
                    if isinstance(first_kv, tuple) and len(first_kv) >= 1:
                        kv_shapes.append(first_kv[0].shape[2])  # seq_len dim
                    else:
                        kv_shapes.append(0)
                else:
                    kv_shapes.append(0)

            max_kv_len = max(kv_shapes) if kv_shapes else 0

            if max_kv_len > 0 and all(s == kv_shapes[0] for s in kv_shapes):
                # All same length — can simply concatenate KV caches
                batched_kv = self._concat_kv_caches([r.kv_cache for r in requests])
                # attention_mask: all 1s (no padding needed)
                attention_mask = torch.ones(
                    batch_size, max_kv_len + 1,
                    dtype=torch.long, device=batched_input.device
                )
            elif max_kv_len > 0:
                # Different lengths — pad KV caches and build attention mask
                batched_kv, attention_mask = self._pad_and_concat_kv_caches(
                    [r.kv_cache for r in requests], kv_shapes, batched_input.device
                )
            else:
                # No KV caches — fall back to sequential
                for req in requests:
                    self._forward_single(req, is_prefill=False)
                return

            # Step 3: Single batched forward pass
            try:
                output = self.model(
                    batched_input,
                    attention_mask=attention_mask,
                    past_key_values=batched_kv,
                    use_cache=True,
                )
                logits = output.logits if hasattr(output, 'logits') else output
                new_kv = getattr(output, 'past_key_values', None)
            except (TypeError, RuntimeError) as e:
                # Model doesn't support batched KV — fall back to sequential
                _logger.debug("Batched decode failed (%s), falling back to sequential", e)
                for req in requests:
                    self._forward_single(req, is_prefill=False)
                return

            # Step 4: Scatter results back to individual requests
            for i, req in enumerate(requests):
                try:
                    req_logits = logits[i:i+1, -1:, :]  # [1, 1, vocab]
                    next_logits = req_logits.squeeze(1)   # [1, vocab]

                    # Unbatch KV cache for this request
                    if new_kv:
                        req.kv_cache = self._unbatch_kv_cache(new_kv, i)
                    else:
                        req.kv_cache = None

                    # Sample next token
                    if req.temperature != 1.0 or req.top_p != 1.0 or req.top_k != 50:
                        next_token = self._sample(next_logits, req.temperature, req.top_k, req.top_p)
                    else:
                        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

                    next_token = next_token.to(req.generated_ids.device)
                    req.generated_ids = torch.cat([req.generated_ids, next_token], dim=-1)
                    req.tokens_generated += 1
                    self._total_tokens_generated += 1

                    # Streaming callback
                    if req.on_token and self.tokenizer:
                        try:
                            token_text = self.tokenizer.decode(
                                [next_token.item()], skip_special_tokens=True
                            )
                            if token_text:
                                req.on_token(token_text)
                        except Exception:
                            pass

                    # Check completion
                    token_id = next_token.item()
                    if token_id == req.stop_token_id or req.tokens_generated >= req.max_new_tokens:
                        self._finish_request_decode(req)

                except Exception as e:
                    _logger.warning("Scatter failed for %s: %s", req.request_id, e)
                    req.status = RequestStatus.ERROR
                    if req.future and not req.future.done():
                        req.future.set_exception(e)

        except Exception as e:
            _logger.warning("Batched decode failed: %s, falling back to sequential", e)
            for req in requests:
                self._forward_single(req, is_prefill=False)

    def _concat_kv_caches(self, kv_list: List[Any]) -> Tuple:
        """Concatenate KV caches from multiple requests (same seq_len)."""
        num_layers = len(kv_list[0])
        result = []
        for layer_idx in range(num_layers):
            keys = torch.cat([kv[layer_idx][0] for kv in kv_list], dim=0)
            values = torch.cat([kv[layer_idx][1] for kv in kv_list], dim=0)
            result.append((keys, values))
        return tuple(result)

    def _pad_and_concat_kv_caches(
        self, kv_list: List[Any], lengths: List[int], device: Any
    ) -> Tuple[Tuple, Any]:
        """Pad KV caches to max length and build attention mask."""
        max_len = max(lengths)
        num_layers = len(kv_list[0])
        batch_size = len(kv_list)

        padded_kv = []
        for layer_idx in range(num_layers):
            padded_keys = []
            padded_values = []
            for i, kv in enumerate(kv_list):
                k, v = kv[layer_idx]
                pad_len = max_len - lengths[i]
                if pad_len > 0:
                    kpad = torch.zeros(
                        k.shape[0], k.shape[1], pad_len, k.shape[3],
                        dtype=k.dtype, device=k.device
                    )
                    vpad = torch.zeros(
                        v.shape[0], v.shape[1], pad_len, v.shape[3],
                        dtype=v.dtype, device=v.device
                    )
                    k = torch.cat([kpad, k], dim=2)  # left-pad
                    v = torch.cat([vpad, v], dim=2)
                padded_keys.append(k)
                padded_values.append(v)
            padded_kv.append((
                torch.cat(padded_keys, dim=0),
                torch.cat(padded_values, dim=0),
            ))

        # Attention mask: 0 for padding, 1 for real tokens + new token
        mask = torch.zeros(batch_size, max_len + 1, dtype=torch.long, device=device)
        for i, l in enumerate(lengths):
            mask[i, max_len - l:] = 1  # real KV positions + new token position
        mask[:, -1] = 1  # current decode token

        return tuple(padded_kv), mask

    def _unbatch_kv_cache(self, batched_kv: Tuple, idx: int) -> Tuple:
        """Extract one request's KV cache from a batched KV cache."""
        result = []
        for layer_kv in batched_kv:
            k, v = layer_kv
            result.append((k[idx:idx+1], v[idx:idx+1]))
        return tuple(result)

    def _forward_single(self, req: InferenceRequest, is_prefill: bool) -> None:
        """Forward pass for a single request with KV cache."""
        try:
            if is_prefill:
                step_input = req.generated_ids
            else:
                step_input = req.generated_ids[:, -1:]

            # Forward with KV cache
            try:
                output = self.model(
                    step_input,
                    past_key_values=req.kv_cache,
                    use_cache=True,
                )
                logits = output.logits if hasattr(output, 'logits') else output
                req.kv_cache = getattr(output, 'past_key_values', None)

                # Store into paged KV cache (async-safe, non-blocking)
                if self.paged_kv and req.kv_cache is not None:
                    try:
                        self.paged_kv.from_hf_cache(req.request_id, req.kv_cache)
                    except Exception:
                        pass
            except (TypeError, AttributeError):
                # Model doesn't support use_cache
                logits = self.model(req.generated_ids)
                if hasattr(logits, 'logits'):
                    logits = logits.logits
                req.kv_cache = None

            # Sample next token
            next_logits = logits[:, -1, :] if logits.dim() >= 2 else logits

            if req.temperature != 1.0 or req.top_p != 1.0 or req.top_k != 50:
                next_token = self._sample(next_logits, req.temperature, req.top_k, req.top_p)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            next_token = next_token.to(req.generated_ids.device)
            req.generated_ids = torch.cat([req.generated_ids, next_token], dim=-1)
            req.tokens_generated += 1
            self._total_tokens_generated += 1

            # Streaming callback
            if req.on_token and self.tokenizer:
                try:
                    token_text = self.tokenizer.decode(
                        [next_token.item()], skip_special_tokens=True
                    )
                    if token_text:
                        req.on_token(token_text)
                except Exception:
                    pass

            # Check completion
            token_id = next_token.item()
            if token_id == req.stop_token_id:
                self._finish_request_decode(req)
            elif req.tokens_generated >= req.max_new_tokens:
                self._finish_request_decode(req)

        except Exception as e:
            _logger.warning("Forward failed for %s: %s", req.request_id, e)
            req.status = RequestStatus.ERROR
            if req.future and not req.future.done():
                req.future.set_exception(e)

    def _sample(
        self,
        logits: Any,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Any:
        """Sample from logits with temperature, top-k, top-p."""
        if temperature > 0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0 and top_k < logits.size(-1):
            topk_vals, _ = torch.topk(logits, top_k)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            logits[logits < threshold] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative probability above threshold
            remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove_mask] = float('-inf')
            # Scatter back
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _finish_request_decode(self, req: InferenceRequest) -> None:
        """Decode and finish a request."""
        if self.tokenizer is None:
            self._finish_request(req, req.prompt)
            return

        text = self.tokenizer.decode(req.generated_ids[0], skip_special_tokens=True)
        self._finish_request(req, text)

    def _finish_request(self, req: InferenceRequest, result: str) -> None:
        """Mark request as finished and resolve its future."""
        req.status = RequestStatus.FINISHED
        req.finished_at = time.time()

        # Free KV cache
        req.kv_cache = None

        # Free paged KV slots if using paged attention
        if self.paged_kv:
            try:
                self.paged_kv.free(req.request_id)
            except Exception:
                pass

        if req.future and not req.future.done():
            req.future.set_result(result)

        _logger.debug("Request %s finished (%d tokens in %.2fs)",
                      req.request_id, req.tokens_generated,
                      (req.finished_at - req.created_at))

    def _evict_completed(self) -> None:
        """Remove finished/errored requests from active batch."""
        still_active = []
        for req in self._active:
            if req.status in (RequestStatus.FINISHED, RequestStatus.ERROR, RequestStatus.CANCELLED):
                self._completed.append(req)
            else:
                still_active.append(req)
        self._active = still_active

        # Trim completed history
        if len(self._completed) > 1000:
            self._completed = self._completed[-500:]


__all__ = [
    "ContinuousBatcher",
    "InferenceRequest",
    "RequestStatus",
]
