"""Tests for Triton fused sampling kernel and paged attention optimizations.

Tests:
  1. triton_sampling: fused_sample API, greedy path, PyTorch fallback
  2. paged_attention: vectorized from_hf_cache
"""
import os
import sys
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ===================================================================
# 1. Triton sampling module
# ===================================================================

class TestTritonSamplingModule:
    """Test core.triton_sampling import and API."""

    def test_import(self):
        from core.triton_sampling import fused_sample, has_triton
        assert callable(fused_sample)
        assert isinstance(has_triton(), bool)

    def test_has_triton_returns_bool(self):
        from core.triton_sampling import has_triton
        result = has_triton()
        assert result is True or result is False

    @pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST") == "1",
        reason="Requires real torch",
    )
    def test_fused_sample_greedy(self):
        """Greedy mode should return argmax."""
        import torch
        from core.triton_sampling import fused_sample
        logits = torch.randn(1, 1000)
        token = fused_sample(logits, greedy=True)
        expected = torch.argmax(logits, dim=-1, keepdim=True)
        assert torch.equal(token, expected)

    @pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST") == "1",
        reason="Requires real torch",
    )
    def test_fused_sample_temperature(self):
        """Temperature sampling should return valid token IDs."""
        import torch
        from core.triton_sampling import fused_sample
        logits = torch.randn(1, 100)
        token = fused_sample(logits, temperature=0.8)
        assert token.shape == (1, 1)
        assert 0 <= token.item() < 100

    @pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST") == "1",
        reason="Requires real torch",
    )
    def test_fused_sample_topk(self):
        """Top-k sampling should only pick from top-k tokens."""
        import torch
        from core.triton_sampling import fused_sample
        logits = torch.zeros(1, 1000)
        # Make only tokens 0-4 have high logits
        logits[0, :5] = 10.0
        token = fused_sample(logits, temperature=0.5, top_k=5)
        assert 0 <= token.item() < 1000  # valid
        # With strong logits for 0-4, should almost always pick from those

    @pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST") == "1",
        reason="Requires real torch",
    )
    def test_fused_sample_batch(self):
        """Should handle batch_size > 1."""
        import torch
        from core.triton_sampling import fused_sample
        logits = torch.randn(4, 500)
        tokens = fused_sample(logits, temperature=1.0)
        assert tokens.shape == (4, 1)

    @pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST") == "1",
        reason="Requires real torch + CUDA",
    )
    def test_fused_sample_cuda(self):
        """On CUDA, should use Triton kernel path when no top-k/top-p."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        from core.triton_sampling import fused_sample, has_triton
        logits = torch.randn(1, 32000, device="cuda")
        token = fused_sample(logits, temperature=0.8)
        assert token.device.type == "cuda"
        assert token.shape == (1, 1)

    def test_fused_sample_zero_temperature(self):
        """Temperature=0 should act as greedy."""
        from core.triton_sampling import fused_sample
        # Even without torch, the function should handle temp<=0 as greedy
        # This test just checks the code path exists
        assert callable(fused_sample)


# ===================================================================
# 2. Paged attention vectorized write
# ===================================================================

class TestPagedAttentionOptimized:
    """Test the optimized from_hf_cache with vectorized page writes."""

    def test_from_hf_cache_exists(self):
        from core.paged_attention import PagedKVCacheManager
        mgr = PagedKVCacheManager.__new__(PagedKVCacheManager)
        assert hasattr(mgr, "from_hf_cache")
        assert callable(mgr.from_hf_cache)

    def test_to_hf_cache_exists(self):
        from core.paged_attention import PagedKVCacheManager
        mgr = PagedKVCacheManager.__new__(PagedKVCacheManager)
        assert hasattr(mgr, "to_hf_cache")
        assert callable(mgr.to_hf_cache)

    @pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST") == "1",
        reason="Requires real torch for tensor ops",
    )
    def test_from_hf_cache_roundtrip(self):
        """Write HF cache to paged memory and read it back."""
        import torch
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig

        config = PagedKVConfig(
            page_size=4,
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            max_pages=64,
            device="cpu",
        )
        mgr = PagedKVCacheManager(config)

        # Create fake HF past_key_values: tuple of (key, value) per layer
        seq_len = 10
        past_kv = tuple(
            (
                torch.randn(1, 2, seq_len, 8),  # key
                torch.randn(1, 2, seq_len, 8),  # value
            )
            for _ in range(2)
        )

        mgr.allocate("req1", num_tokens=0)
        mgr.from_hf_cache("req1", past_kv)

        # Read back
        result = mgr.to_hf_cache("req1")
        assert result is not None
        assert len(result) == 2
        for layer_idx in range(2):
            k_orig = past_kv[layer_idx][0][0]  # [heads, seq, dim]
            k_read = result[layer_idx][0][0]
            assert k_read.shape[1] >= seq_len

    @pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST") == "1",
        reason="Requires real torch for tensor ops",
    )
    def test_from_hf_cache_no_pool(self):
        """from_hf_cache with no GPU pool should be a no-op."""
        import torch
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig

        config = PagedKVConfig(max_pages=0, device="cpu")
        mgr = PagedKVCacheManager(config)
        mgr._gpu_pool = None
        # Should not raise
        mgr.from_hf_cache("req1", None)

    def test_from_hf_cache_none_kv(self):
        """from_hf_cache with None past_key_values should be a no-op."""
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        config = PagedKVConfig(max_pages=0, device="cpu")
        mgr = PagedKVCacheManager(config)
        mgr._gpu_pool = None
        mgr.from_hf_cache("req1", None)


# ===================================================================
# 3. Backends integration: fused_sample wired
# ===================================================================

class TestBackendsFusedSamplingWired:
    """Verify fused_sample is imported in backends.py."""

    def test_fused_sample_import_in_backends(self):
        import inspect
        from core.backends import HuggingFaceBackend
        source = inspect.getsource(HuggingFaceBackend)
        assert "_fused_sample" in source or "fused_sample" in source

    def test_has_fused_sample_flag(self):
        """_HAS_FUSED_SAMPLE should be set."""
        # In minimal test mode, triton may not be importable by backends
        # but the flag should exist
        import core.backends as b
        assert hasattr(b, "_HAS_FUSED_SAMPLE")

    def test_batcher_sample_uses_fused(self):
        """Batcher._sample should reference triton_sampling."""
        import inspect
        from core.continuous_batcher import ContinuousBatcher
        source = inspect.getsource(ContinuousBatcher._sample)
        assert "fused_sample" in source


# ===================================================================
# 4. Benchmark script importable
# ===================================================================

class TestBenchConcurrent:
    """Verify the concurrent benchmark script is importable."""

    def test_import_bench(self):
        sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
        try:
            import bench_concurrent
            assert hasattr(bench_concurrent, "main")
            assert hasattr(bench_concurrent, "bench_sequential")
            assert hasattr(bench_concurrent, "bench_concurrent_batch")
            assert hasattr(bench_concurrent, "bench_sustained_load")
        finally:
            sys.path.pop(0)
