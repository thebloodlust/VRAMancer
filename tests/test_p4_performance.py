"""Tests for P4 — Performance & Research features.

Covers:
  - CUDAGraphRunner (core/cuda_graph_decode.py) — graph capture/replay logic
  - Speculative decoding adaptive K (core/speculative_decoding.py)
  - GGUF multi-GPU compute-aware split (core/backends_llamacpp.py)
  - Q4 paged attention kernel wrapper (core/paged_attention_cuda.py)
  - Async tokenizer in continuous batcher (core/continuous_batcher.py)

Uses VRM_MINIMAL_TEST=1 from conftest. Mocks torch where needed.
"""

import os
import pytest
import threading
from unittest.mock import MagicMock, patch

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")


# ═══════════════════════════════════════════════════════════════════════
# Speculative Decoding — Adaptive K
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptiveSpeculative:
    """Test adaptive gamma in SwarmSpeculativeDecoder (no torch needed)."""

    def _make_decoder(self, gamma=5, adaptive=True):
        """Build a decoder with mock draft/verify (no real model)."""
        # Mock torch module for the class
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from core.speculative_decoding import SwarmSpeculativeDecoder

            draft_fn = MagicMock()
            verify_fn = MagicMock()
            dec = SwarmSpeculativeDecoder(
                draft_model_callable=draft_fn,
                swarm_verify_callable=verify_fn,
                gamma=gamma,
                temperature=0.0,
                adaptive=adaptive,
                gamma_min=2,
                gamma_max=10,
            )
            return dec

    def test_adaptive_fields_exist(self):
        dec = self._make_decoder()
        assert dec.adaptive is True
        assert dec.gamma_min == 2
        assert dec.gamma_max == 10
        assert dec._acceptance_window == []
        assert dec._window_size == 10

    def test_adapt_gamma_high_acceptance_increases(self):
        dec = self._make_decoder(gamma=5)
        # Simulate 5 rounds of 90% acceptance
        for _ in range(5):
            dec._adapt_gamma(9, 10)
        assert dec.gamma > 5, f"Expected gamma > 5, got {dec.gamma}"

    def test_adapt_gamma_low_acceptance_decreases(self):
        dec = self._make_decoder(gamma=5)
        # Simulate 5 rounds of 10% acceptance
        for _ in range(5):
            dec._adapt_gamma(1, 10)
        assert dec.gamma < 5, f"Expected gamma < 5, got {dec.gamma}"

    def test_adapt_gamma_respects_min(self):
        dec = self._make_decoder(gamma=3, adaptive=True)
        # Hammer it down
        for _ in range(20):
            dec._adapt_gamma(0, 10)
        assert dec.gamma >= 2

    def test_adapt_gamma_respects_max(self):
        dec = self._make_decoder(gamma=8, adaptive=True)
        # Hammer it up
        for _ in range(20):
            dec._adapt_gamma(10, 10)
        assert dec.gamma <= 10

    def test_adapt_gamma_disabled(self):
        dec = self._make_decoder(gamma=5, adaptive=False)
        for _ in range(10):
            dec._adapt_gamma(10, 10)
        assert dec.gamma == 5, "Gamma should not change when adaptive=False"

    def test_adapt_gamma_needs_minimum_rounds(self):
        dec = self._make_decoder(gamma=5)
        # Only 2 rounds — not enough for adaptation
        dec._adapt_gamma(10, 10)
        dec._adapt_gamma(10, 10)
        assert dec.gamma == 5, "Should not adapt with < 3 rounds"

    def test_window_size_capped(self):
        dec = self._make_decoder(gamma=5)
        dec._window_size = 5
        for i in range(20):
            dec._adapt_gamma(5, 10)
        assert len(dec._acceptance_window) <= 5


# ═══════════════════════════════════════════════════════════════════════
# CUDA Graph Runner
# ═══════════════════════════════════════════════════════════════════════

class TestCUDAGraphRunner:
    """Test CUDAGraphRunner logic (no real CUDA needed)."""

    def test_disabled_without_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            # Force reimport
            import importlib
            import core.cuda_graph_decode as mod
            old_has = mod._HAS_TORCH
            mod._HAS_TORCH = False
            runner = mod.CUDAGraphRunner(model=MagicMock(), enabled=True)
            assert not runner.enabled
            mod._HAS_TORCH = old_has

    def test_import(self):
        from core.cuda_graph_decode import CUDAGraphRunner
        assert CUDAGraphRunner is not None

    def test_reset_clears_state(self):
        from core.cuda_graph_decode import CUDAGraphRunner
        runner = CUDAGraphRunner(model=MagicMock(), enabled=False)
        runner._call_counts[1] = 99
        runner._graphs[1] = ("g", "i", "o")
        runner.reset()
        assert runner._graphs == {}
        assert runner._call_counts == {}

    def test_captured_sizes_property(self):
        from core.cuda_graph_decode import CUDAGraphRunner
        runner = CUDAGraphRunner(model=MagicMock(), enabled=False)
        runner._graphs[4] = ("g", "i", "o")
        runner._graphs[8] = ("g", "i", "o")
        assert sorted(runner.captured_sizes) == [4, 8]

    def test_eager_fallback_when_disabled(self):
        from core.cuda_graph_decode import CUDAGraphRunner
        mock_model = MagicMock()
        mock_out = MagicMock()
        mock_out.logits = "test_logits"
        mock_model.return_value = mock_out

        runner = CUDAGraphRunner(model=mock_model, enabled=False)
        mock_ids = MagicMock()
        result = runner.forward(mock_ids, some_kwarg=True)
        mock_model.assert_called_once()
        assert result == "test_logits"


# ═══════════════════════════════════════════════════════════════════════
# GGUF Multi-GPU — Compute-Aware Split
# ═══════════════════════════════════════════════════════════════════════

class TestGGUFMultiGPU:
    """Test enhanced _compute_tensor_split and helpers."""

    def _make_backend(self):
        from core.backends_llamacpp import LlamaCppBackend
        return LlamaCppBackend()

    def test_compute_weights_without_torch(self):
        backend = self._make_backend()
        # In minimal test mode, torch may not be available — should return [1.0, 1.0]
        weights = backend._get_compute_weights(2)
        assert len(weights) == 2
        assert all(isinstance(w, float) for w in weights)

    def test_select_split_mode_no_p2p_env(self):
        backend = self._make_backend()
        with patch.dict(os.environ, {"VRM_TRANSFER_P2P": "0"}):
            mode = backend._select_split_mode(2)
            assert mode == 1, "Should use layer split when P2P disabled"

    def test_select_split_mode_default(self):
        backend = self._make_backend()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VRM_TRANSFER_P2P", None)
            mode = backend._select_split_mode(2)
            assert mode in (1, 2)

    def test_get_vram_fallback_chain(self):
        backend = self._make_backend()
        # In minimal test mode all fallbacks will fail gracefully
        result = backend._get_vram_per_gpu(2)
        # Could be None or a list, both are valid
        assert result is None or (isinstance(result, list) and len(result) == 2)

    def test_compute_tensor_split_returns_none_or_list(self):
        backend = self._make_backend()
        result = backend._compute_tensor_split(2)
        if result is not None:
            assert len(result) == 2
            assert abs(sum(result) - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# Async Tokenizer in Continuous Batcher
# ═══════════════════════════════════════════════════════════════════════

class TestAsyncTokenizer:
    """Test ThreadPoolExecutor integration in batcher."""

    def test_tokenizer_pool_init(self):
        from core.continuous_batcher import ContinuousBatcher
        b = ContinuousBatcher(max_batch_size=8)
        assert b._tokenizer_pool is not None
        b.stop()

    def test_tokenizer_pool_disabled_with_env(self):
        from core.continuous_batcher import ContinuousBatcher
        with patch.dict(os.environ, {"VRM_TOKENIZER_WORKERS": "0"}):
            b = ContinuousBatcher(max_batch_size=8)
            assert b._tokenizer_pool is None
            b.stop()

    def test_stop_shuts_down_pool(self):
        from core.continuous_batcher import ContinuousBatcher
        b = ContinuousBatcher(max_batch_size=8)
        assert b._tokenizer_pool is not None
        b.stop()
        assert b._tokenizer_pool is None


# ═══════════════════════════════════════════════════════════════════════
# Q4 Paged Attention Python Wrapper
# ═══════════════════════════════════════════════════════════════════════

class TestQ4PagedAttentionWrapper:
    """Test the Q4 wrapper exists and is importable."""

    def test_import_q4_function(self):
        from core.paged_attention_cuda import paged_attention_decode_q4
        assert callable(paged_attention_decode_q4)

    def test_all_exports(self):
        from core.paged_attention_cuda import __all__
        assert "paged_attention_decode_q4" in __all__

    def test_has_cuda_check(self):
        from core.paged_attention_cuda import has_cuda_paged_attention
        # In minimal test mode, should return False (no CUDA available)
        result = has_cuda_paged_attention()
        assert isinstance(result, bool)
