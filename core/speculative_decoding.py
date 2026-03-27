"""
Swarm Speculative Decoding
==========================

Speculative Decoding for distributed inference: a small draft model
generates N tokens ahead, the full verifier validates them in one
forward pass. Correctly guessed tokens are free; wrong ones get
corrected at position K and re-drafted.

Supports:
  - Greedy (temperature=0) and stochastic (temperature>0) verification
  - Auto-creation of a draft model from the same backend (scaled down)
  - Prometheus metrics export
  - Batch size 1 (per-request)
"""

import os
import time
import logging
from typing import Any, Callable, Optional

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    from core.metrics import INFER_REQUESTS
    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False

logger = logging.getLogger("vramancer.speculative")


# ── Prometheus metrics (lazy) ──────────────────────────────────────────
_SPEC_DRAFTED = None
_SPEC_ACCEPTED = None
_SPEC_ROUNDS = None

def _init_spec_metrics():
    global _SPEC_DRAFTED, _SPEC_ACCEPTED, _SPEC_ROUNDS
    if _SPEC_DRAFTED is not None:
        return
    try:
        from prometheus_client import Counter
        _SPEC_DRAFTED = Counter(
            "vramancer_speculative_drafted_total",
            "Tokens drafted by speculative decoder",
        )
        _SPEC_ACCEPTED = Counter(
            "vramancer_speculative_accepted_total",
            "Tokens accepted by verifier",
        )
        _SPEC_ROUNDS = Counter(
            "vramancer_speculative_rounds_total",
            "Speculative decoding rounds",
        )
    except Exception:
        pass


# ── Draft model mapping ───────────────────────────────────────────────
# Maps main model family prefixes to recommended small draft models.
# Used when VRM_DRAFT_MODEL is not set — picks the best small model
# from the same family so the tokenizer is compatible.
_DRAFT_MODEL_MAP = {
    "Qwen/Qwen2.5-": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2-": "Qwen/Qwen2-0.5B-Instruct",
    "meta-llama/Llama-3": "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-2": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistralai/": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2": "distilbert/distilgpt2",
    "openai-community/gpt2": "distilbert/distilgpt2",
    "TinyLlama/": "distilbert/distilgpt2",
}


def _guess_draft_model(main_model_name: str) -> Optional[str]:
    """Return a sensible draft model name for a given main model, or None."""
    if not main_model_name:
        return None
    for prefix, draft in _DRAFT_MODEL_MAP.items():
        if main_model_name.startswith(prefix):
            return draft
    return None


# ── Draft model factory ───────────────────────────────────────────────

def create_draft_callable(
    backend: Any,
    draft_model_name: Optional[str] = None,
    main_model_name: Optional[str] = None,
) -> Optional[Callable]:
    """Create a draft_model_callable from a loaded HuggingFaceBackend.

    If *draft_model_name* is provided, loads that model as the drafter
    (e.g. ``"distilgpt2"`` for a GPT-2 main model).

    If not provided, tries to auto-detect a suitable small draft model
    from ``_DRAFT_MODEL_MAP`` based on *main_model_name*.

    Self-drafting (reusing the main model) is NOT used — it provides
    no speedup since the draft phase runs the full model N times.

    Returns ``None`` when no suitable drafter can be built.
    """
    if _MINIMAL or not _HAS_TORCH:
        return None

    # Resolve draft model name
    if not draft_model_name and main_model_name:
        draft_model_name = _guess_draft_model(main_model_name)
        if draft_model_name:
            logger.info("Auto-selected draft model: %s for main model: %s",
                        draft_model_name, main_model_name)

    if draft_model_name:
        return _load_external_draft_model(draft_model_name, backend)

    # No draft model available — self-drafting (same model) gives no speedup
    logger.debug("No draft model specified and no auto-mapping found — "
                 "speculative decoding disabled")
    return None


def _load_external_draft_model(
    model_name: str, backend: Any,
) -> Optional[Callable]:
    """Load a small external model as the drafter."""
    try:
        from transformers import AutoModelForCausalLM
        device = "cpu"
        if _HAS_TORCH and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        draft = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
        ).to(device).eval()
        logger.info(f"Loaded external draft model: {model_name} on {device}")

        def _external_draft(input_ids: "torch.Tensor", num_tokens: int) -> "torch.Tensor":
            ids = input_ids.to(device)
            tokens = []
            for _ in range(num_tokens):
                with torch.no_grad():
                    out = draft(ids)
                    logits = out.logits if hasattr(out, "logits") else out[0]
                    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    tokens.append(next_id)
                    ids = torch.cat([ids, next_id], dim=-1)
            return torch.cat(tokens, dim=-1).to(input_ids.device)

        return _external_draft
    except Exception as exc:
        logger.warning(f"Cannot load draft model {model_name}: {exc}")
        return None


# ── Main decoder ──────────────────────────────────────────────────────

class SwarmSpeculativeDecoder:
    """Speculative decoding engine for VRAMancer inference pipeline."""

    def __init__(
        self,
        draft_model_callable: Callable,
        swarm_verify_callable: Callable,
        gamma: int = 5,
        temperature: float = 0.0,
        adaptive: bool = True,
        gamma_min: int = 2,
        gamma_max: int = 12,
    ):
        self.draft_model = draft_model_callable
        self.swarm_verify = swarm_verify_callable
        self.gamma = gamma
        self.temperature = temperature
        self.log = logger

        # Adaptive K: adjust gamma based on rolling acceptance rate
        self.adaptive = adaptive
        self.gamma_min = max(1, gamma_min)
        self.gamma_max = gamma_max
        self._acceptance_window: list = []  # last N (accepted, drafted) tuples
        self._window_size = int(os.environ.get("VRM_SPEC_WINDOW", "10"))

        self.total_drafted = 0
        self.total_accepted = 0
        self.latency_saved_ms = 0.0

        _init_spec_metrics()

    def _sample_token(self, logits: "torch.Tensor") -> "torch.Tensor":
        """Sample a single token from logits respecting temperature."""
        if self.temperature <= 0.0:
            return logits.argmax(dim=-1)
        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _adapt_gamma(self, accepted: int, drafted: int) -> None:
        """Adjust gamma based on rolling acceptance rate.

        High acceptance (>80%) → increase gamma (speculatively guess more)
        Low acceptance  (<30%) → decrease gamma (waste less compute)
        """
        if not self.adaptive:
            return

        self._acceptance_window.append((accepted, drafted))
        if len(self._acceptance_window) > self._window_size:
            self._acceptance_window.pop(0)

        if len(self._acceptance_window) < 3:
            return  # need at least 3 rounds of data

        total_a = sum(a for a, _ in self._acceptance_window)
        total_d = sum(d for _, d in self._acceptance_window)
        rate = total_a / max(1, total_d)

        old_gamma = self.gamma
        if rate > 0.80:
            self.gamma = min(self.gamma + 1, self.gamma_max)
        elif rate < 0.30:
            self.gamma = max(self.gamma - 1, self.gamma_min)

        if self.gamma != old_gamma:
            self.log.debug("[Speculative] Adaptive K: %d → %d (rate=%.0f%%)",
                           old_gamma, self.gamma, rate * 100)

    @torch.no_grad()
    def generate(self, input_ids: "torch.Tensor", max_new_tokens: int) -> "torch.Tensor":
        """Run the speculative decoding loop."""
        start_time = time.perf_counter()
        generated_ids = input_ids.clone()
        tokens_yielded = 0

        self.log.info(f"[Speculative] gamma={self.gamma}, temp={self.temperature}")

        while tokens_yielded < max_new_tokens:
            # 1. Draft — fast local model predicts gamma tokens
            draft_tokens = self.draft_model(generated_ids, self.gamma)

            # The speculated sequence
            speculated_ids = torch.cat([generated_ids, draft_tokens], dim=-1)

            # 2. Verify — heavy model scores all draft tokens in one pass
            swarm_start = time.perf_counter()
            target_logits = self.swarm_verify(speculated_ids)

            # 3. Accept/reject — compare draft vs verifier
            accepted_count = 0
            verify_logits = target_logits[:, -self.gamma - 1:-1, :]

            if self.temperature <= 0.0:
                # Greedy: exact match
                target_tokens = verify_logits.argmax(dim=-1)
                for i in range(self.gamma):
                    if draft_tokens[0, i] == target_tokens[0, i]:
                        accepted_count += 1
                    else:
                        break
            else:
                # Stochastic: acceptance probability = min(1, q(x)/p(x))
                for i in range(self.gamma):
                    draft_id = draft_tokens[0, i].item()
                    v_logits = verify_logits[0, i]
                    q_prob = torch.softmax(v_logits / self.temperature, dim=-1)
                    # Accept with probability proportional to verifier confidence
                    q_val = q_prob[draft_id].item()
                    # Simple threshold: accept if verifier top-1 agrees
                    top_id = v_logits.argmax(dim=-1).item()
                    if draft_id == top_id or q_val > 0.5:
                        accepted_count += 1
                    else:
                        break

            # Accept matched tokens
            if accepted_count > 0:
                generated_ids = torch.cat(
                    [generated_ids, draft_tokens[:, :accepted_count]], dim=-1
                )

            # 4. Correction — always get at least one guaranteed token
            correction_pos = -(self.gamma - accepted_count + 1)
            correction_logits = target_logits[:, correction_pos, :]
            correction_token = self._sample_token(correction_logits).unsqueeze(0)
            if correction_token.dim() == 1:
                correction_token = correction_token.unsqueeze(0)
            generated_ids = torch.cat([generated_ids, correction_token], dim=-1)

            tokens_to_add = accepted_count + 1
            tokens_yielded += tokens_to_add

            # Telemetry
            self.total_drafted += self.gamma
            self.total_accepted += accepted_count

            # Adaptive K — adjust gamma for next round
            self._adapt_gamma(accepted_count, self.gamma)

            swarm_elapsed = time.perf_counter() - swarm_start
            naive_time = swarm_elapsed * (accepted_count + 1)
            self.latency_saved_ms += max(0, naive_time - swarm_elapsed) * 1000

            if _SPEC_DRAFTED:
                _SPEC_DRAFTED.inc(self.gamma)
            if _SPEC_ACCEPTED:
                _SPEC_ACCEPTED.inc(accepted_count)
            if _SPEC_ROUNDS:
                _SPEC_ROUNDS.inc()

            self.log.debug(
                f"[Speculative] Drafted {self.gamma} | "
                f"Accepted {accepted_count} | Corrected 1 "
                f"({accepted_count / self.gamma * 100:.0f}%)"
            )

        rate = self.total_accepted / max(1, self.total_drafted) * 100
        self.log.info(
            f"[Speculative] Done: {self.total_accepted}/{self.total_drafted} "
            f"accepted ({rate:.1f}%), saved {self.latency_saved_ms:.0f}ms"
        )
        return generated_ids

