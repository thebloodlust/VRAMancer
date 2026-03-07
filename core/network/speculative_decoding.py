"""VRAMancer Speculative Decoding Asymétrique.

Implements speculative decoding between a small Draft model and a large
Target model, potentially on different nodes of a heterogeneous cluster.

Algorithm (Leviathan et al., 2023):
  1. Draft model generates gamma tokens quickly (small model, fast node)
  2. Target model validates all gamma tokens in ONE forward pass (large model)
  3. Accept matching prefix, reject divergent suffix
  4. Target provides the corrected token at the rejection point
  5. Expected accepted tokens per step: ~gamma * acceptance_rate

This implementation uses InferencePipeline instances for both models,
enabling real model-based draft/validation rather than simulation.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Callable, List, Optional, Tuple

_logger = logging.getLogger("vramancer.speculative")

try:
    import torch
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    _TORCH = False

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")


class SpeculativeDecoder:
    """Speculative decoding with a draft and target model.

    Both models can be local InferencePipeline instances or remote
    (accessed via RPC through the cluster API). The decoder controls
    the speculative loop: draft → validate → accept/reject → correct.

    Parameters
    ----------
    draft_model_id : str
        Model name for the small/fast draft model.
    target_model_id : str
        Model name for the large/accurate target model.
    gamma : int
        Number of speculative tokens per iteration (default 5).
    draft_pipeline : Any
        InferencePipeline (or compatible) for the draft model.
        If None, one will be created lazily.
    target_pipeline : Any
        InferencePipeline (or compatible) for the target model.
        If None, one will be created lazily.
    """

    def __init__(
        self,
        draft_model_id: str,
        target_model_id: str,
        gamma: int = 5,
        draft_pipeline: Any = None,
        target_pipeline: Any = None,
    ):
        self.draft_model_id = draft_model_id
        self.target_model_id = target_model_id
        self.gamma = gamma
        self._draft = draft_pipeline
        self._target = target_pipeline
        self._tokenizer = None

        # Stats
        self._total_accepted = 0
        self._total_drafted = 0
        self._total_iterations = 0

    def _ensure_pipelines(self) -> None:
        """Lazily initialize pipelines if not provided."""
        if self._draft is None or self._target is None:
            from core.inference_pipeline import InferencePipeline
            if self._draft is None:
                self._draft = InferencePipeline()
                self._draft.load(self.draft_model_id)
            if self._target is None:
                self._target = InferencePipeline()
                self._target.load(self.target_model_id)
        # Get tokenizer from target (assumed to be the reference)
        if self._tokenizer is None:
            backend = getattr(self._target, 'backend', None)
            if backend:
                self._tokenizer = getattr(backend, 'tokenizer', None)

    def generate(self, prompt: str, max_tokens: int = 50, **kwargs) -> str:
        """Generate text using speculative decoding.

        Falls back to a simple target-model generation if torch
        is unavailable or in minimal test mode.
        """
        if _MINIMAL or not _TORCH:
            # Stub mode: delegate to target pipeline's generate
            self._ensure_pipelines()
            return self._target.generate(prompt, max_new_tokens=max_tokens)

        self._ensure_pipelines()
        tokenizer = self._tokenizer
        if tokenizer is None:
            _logger.warning("No tokenizer available, falling back to direct generation")
            return self._target.generate(prompt, max_new_tokens=max_tokens)

        # Tokenize prompt
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        generated = input_ids
        eos_id = getattr(tokenizer, 'eos_token_id', None)

        tokens_generated = 0
        while tokens_generated < max_tokens:
            # 1. Draft: generate gamma tokens greedily
            draft_ids, draft_logits = self._draft_tokens(generated, self.gamma)
            self._total_drafted += draft_ids.shape[-1]
            self._total_iterations += 1

            # 2. Validate: target model scores ALL draft tokens in one pass
            accepted_len = self._validate_draft(
                generated, draft_ids, draft_logits
            )
            self._total_accepted += accepted_len

            # 3. Accept matching prefix
            if accepted_len > 0:
                generated = torch.cat(
                    [generated, draft_ids[:, :accepted_len]], dim=-1
                )
                tokens_generated += accepted_len

            # 4. If rejected, get correction from target
            if accepted_len < draft_ids.shape[-1]:
                correction_id = self._get_correction_token(generated)
                generated = torch.cat(
                    [generated, correction_id.unsqueeze(0).unsqueeze(0)], dim=-1
                )
                tokens_generated += 1

            # Check EOS
            if eos_id is not None and generated[0, -1].item() == eos_id:
                break

        return tokenizer.decode(generated[0], skip_special_tokens=True)

    def _draft_tokens(
        self, context: Any, num_tokens: int
    ) -> Tuple[Any, List[Any]]:
        """Generate num_tokens from the draft model autoregressively.

        Returns (token_ids [1, num_tokens], per-step logits list).
        """
        draft_ids = []
        draft_logits = []
        current = context

        for _ in range(num_tokens):
            logits = self._draft.backend.infer(current)
            if isinstance(logits, tuple):
                logits = logits[0]
            # Take last position
            last_logits = logits[:, -1:, :] if logits.dim() == 3 else logits
            next_token = torch.argmax(last_logits.squeeze(1), dim=-1)
            draft_ids.append(next_token)
            draft_logits.append(last_logits.squeeze(1))
            current = torch.cat(
                [current, next_token.unsqueeze(-1)], dim=-1
            )

        return torch.stack(draft_ids, dim=-1), draft_logits

    def _validate_draft(
        self, context: Any, draft_ids: Any, draft_logits: List[Any]
    ) -> int:
        """Validate draft tokens with the target model.

        Runs a single forward pass on context + draft tokens,
        then checks where target and draft agree.

        Returns number of accepted tokens (0 to gamma).
        """
        # Build full sequence: context + all draft tokens
        full_input = torch.cat([context, draft_ids], dim=-1)

        # Target model: single forward pass on the full sequence
        target_logits = self._target.backend.infer(full_input)
        if isinstance(target_logits, tuple):
            target_logits = target_logits[0]

        # Compare target predictions with draft tokens
        # For position i in [context_len, context_len + gamma),
        # target predicts token at i+1, draft proposed draft_ids[i-context_len]
        context_len = context.shape[-1]
        accepted = 0

        for i in range(draft_ids.shape[-1]):
            pos = context_len + i - 1  # logits position that predicts next token
            if pos < 0 or pos >= target_logits.shape[1]:
                break

            target_token = torch.argmax(target_logits[:, pos, :], dim=-1)
            draft_token = draft_ids[:, i]

            if target_token.item() == draft_token.item():
                accepted += 1
            else:
                break

        return accepted

    def _get_correction_token(self, context: Any) -> Any:
        """Get the target model's prediction for the next token."""
        logits = self._target.backend.infer(context)
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits[:, -1, :], dim=-1)

    @property
    def acceptance_rate(self) -> float:
        """Average fraction of draft tokens accepted."""
        if self._total_drafted == 0:
            return 0.0
        return self._total_accepted / self._total_drafted

    def stats(self) -> dict:
        """Return speculative decoding statistics."""
        return {
            "total_iterations": self._total_iterations,
            "total_drafted": self._total_drafted,
            "total_accepted": self._total_accepted,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "gamma": self.gamma,
            "draft_model": self.draft_model_id,
            "target_model": self.target_model_id,
        }

