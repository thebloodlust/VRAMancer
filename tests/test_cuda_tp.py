"""Tests for PagedAttention CUDA kernel and Tensor Parallelism."""
import os
import sys
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
    _NUM_GPUS = torch.cuda.device_count() if _HAS_CUDA else 0
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False
    _NUM_GPUS = 0

skip_no_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
skip_no_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
skip_no_multi_gpu = pytest.mark.skipif(_NUM_GPUS < 2, reason="Need 2+ GPUs")
skip_minimal = pytest.mark.skipif(bool(_MINIMAL), reason="VRM_MINIMAL_TEST set")


# ================================================================
# PagedAttention CUDA kernel tests
# ================================================================

class TestPagedAttentionCUDA:
    """Test the custom CUDA paged attention kernel + PyTorch fallback."""

    def test_import(self):
        from core.paged_attention_cuda import paged_attention_decode, has_cuda_paged_attention
        assert callable(paged_attention_decode)
        assert callable(has_cuda_paged_attention)

    def test_pytorch_fallback_import(self):
        from core.paged_attention_cuda import _pytorch_paged_attention_decode
        assert callable(_pytorch_paged_attention_decode)

    @skip_no_cuda
    @skip_minimal
    def test_pytorch_fallback_gpt2(self):
        """PyTorch fallback correctness with GPT-2-like dims."""
        from core.paged_attention_cuda import _pytorch_paged_attention_decode
        B, H, D = 2, 12, 64
        PS, L = 16, 6
        ctx = 48  # 3 pages
        pages_per = ctx // PS
        max_pages = B * pages_per + 4

        q = torch.randn(B, H, D, device="cuda", dtype=torch.float32)
        pool = torch.randn(max_pages, L, 2, H, PS, D, device="cuda", dtype=torch.float16)
        pt = torch.full((B, pages_per + 1), -1, device="cuda", dtype=torch.int32)
        for b in range(B):
            for p in range(pages_per):
                pt[b, p] = b * pages_per + p
        ctx_t = torch.full((B,), ctx, device="cuda", dtype=torch.int32)

        out = _pytorch_paged_attention_decode(q, pool, pt, ctx_t, 0, 1.0 / math.sqrt(D))
        assert out.shape == (B, H, D)
        assert out.dtype == torch.float32
        assert not torch.isnan(out).any()

    @skip_no_cuda
    @skip_minimal
    def test_pytorch_fallback_gqa(self):
        """PyTorch fallback with GQA (32 heads, 8 KV heads)."""
        from core.paged_attention_cuda import _pytorch_paged_attention_decode
        B, H, KH, D = 1, 32, 8, 128
        PS, L = 16, 1
        ctx = 64

        q = torch.randn(B, H, D, device="cuda", dtype=torch.float32)
        pool = torch.randn(8, L, 2, KH, PS, D, device="cuda", dtype=torch.float16)
        pt = torch.arange(4, device="cuda", dtype=torch.int32).unsqueeze(0)
        ctx_t = torch.tensor([ctx], device="cuda", dtype=torch.int32)

        out = _pytorch_paged_attention_decode(q, pool, pt, ctx_t, 0, 1.0 / math.sqrt(D))
        assert out.shape == (B, H, D)

    @skip_no_cuda
    @skip_minimal
    def test_paged_attention_dispatch(self):
        """Test the main dispatch function (CUDA kernel or fallback)."""
        from core.paged_attention_cuda import paged_attention_decode
        B, H, D = 1, 12, 64
        PS, L = 16, 4
        ctx = 32

        q = torch.randn(B, H, D, device="cuda", dtype=torch.float32)
        pool = torch.randn(4, L, 2, H, PS, D, device="cuda", dtype=torch.float16)
        pt = torch.arange(2, device="cuda", dtype=torch.int32).unsqueeze(0)
        ctx_t = torch.tensor([ctx], device="cuda", dtype=torch.int32)

        out = paged_attention_decode(q, pool, pt, ctx_t, 0)
        assert out.shape == (B, H, D)
        assert not torch.isnan(out).any()

    @skip_no_cuda
    @skip_minimal
    def test_zero_context_length(self):
        """Zero context length returns zeros."""
        from core.paged_attention_cuda import _pytorch_paged_attention_decode
        B, H, D = 1, 4, 64
        q = torch.randn(B, H, D, device="cuda")
        pool = torch.randn(2, 1, 2, H, 16, D, device="cuda", dtype=torch.float16)
        pt = torch.zeros(B, 1, device="cuda", dtype=torch.int32)
        ctx_t = torch.zeros(B, device="cuda", dtype=torch.int32)

        out = _pytorch_paged_attention_decode(q, pool, pt, ctx_t, 0, 0.125)
        assert out.shape == (B, H, D)
        assert (out == 0).all()

    @skip_no_cuda
    @skip_minimal
    def test_single_token_context(self):
        """Single token in a single page."""
        from core.paged_attention_cuda import _pytorch_paged_attention_decode
        B, H, D = 1, 4, 64
        PS = 16
        q = torch.randn(B, H, D, device="cuda")
        pool = torch.randn(2, 1, 2, H, PS, D, device="cuda", dtype=torch.float16)
        pt = torch.tensor([[0]], device="cuda", dtype=torch.int32)
        ctx_t = torch.tensor([1], device="cuda", dtype=torch.int32)

        out = _pytorch_paged_attention_decode(q, pool, pt, ctx_t, 0, 1.0 / math.sqrt(D))
        assert out.shape == (B, H, D)
        assert not torch.isnan(out).any()


class TestPagedAttentionManagerIntegration:
    """Test compute_attention_decode on PagedKVCacheManager."""

    def test_compute_attention_method_exists(self):
        from core.paged_attention import PagedKVCacheManager
        mgr = PagedKVCacheManager()
        assert hasattr(mgr, "compute_attention_decode")


# ================================================================
# Tensor Parallelism tests
# ================================================================

class TestTensorParallel:
    """Test tensor parallelism module."""

    def test_import(self):
        from core.tensor_parallel import apply_tensor_parallel, TPModel
        assert callable(apply_tensor_parallel)

    @skip_no_torch
    def test_shard_column(self):
        from core.tensor_parallel import _shard_column
        w = torch.randn(8, 4)
        shards = _shard_column(w, 2, dim=0)
        assert len(shards) == 2
        assert shards[0].shape == (4, 4)
        # Recombine
        assert torch.allclose(torch.cat(shards, dim=0), w)

    @skip_no_torch
    def test_shard_row(self):
        from core.tensor_parallel import _shard_row
        w = torch.randn(4, 8)
        shards = _shard_row(w, 2, dim=1)
        assert len(shards) == 2
        assert shards[0].shape == (4, 4)
        assert torch.allclose(torch.cat(shards, dim=1), w)

    @skip_no_torch
    def test_detect_architecture_llama(self):
        from core.tensor_parallel import _detect_architecture

        class FakeConfig:
            model_type = "llama"
        class FakeModel:
            config = FakeConfig()
            def named_modules(self):
                return []

        assert _detect_architecture(FakeModel()) == "llama"

    @skip_no_torch
    def test_detect_architecture_gpt2(self):
        from core.tensor_parallel import _detect_architecture

        class FakeConfig:
            model_type = "gpt2"
        class FakeModel:
            config = FakeConfig()
            def named_modules(self):
                return []

        assert _detect_architecture(FakeModel()) == "gpt2"

    @skip_no_multi_gpu
    @skip_minimal
    def test_nccl_all_reduce(self):
        """NCCL all-reduce sum works across 2 GPUs."""
        from core.tensor_parallel import _nccl_all_reduce
        t0 = torch.ones(4, 4, device="cuda:0") * 3
        t1 = torch.ones(4, 4, device="cuda:1") * 5
        _nccl_all_reduce([t0, t1])
        assert torch.allclose(t0, torch.full_like(t0, 8.0))
        assert torch.allclose(t1, torch.full_like(t1, 8.0))

    @skip_no_torch
    def test_tp_linear(self):
        from core.tensor_parallel import TPLinear
        w = torch.randn(4, 8)
        b = torch.randn(4)
        tpl = TPLinear(w, b, "cpu")
        x = torch.randn(2, 8)
        out = tpl(x)
        assert out.shape == (2, 4)

    @skip_no_torch
    def test_tpmodel_requires_2_devices(self):
        from core.tensor_parallel import apply_tensor_parallel

        class DummyModel:
            pass

        with pytest.raises(ValueError, match="at least 2"):
            apply_tensor_parallel(DummyModel(), devices=["cuda:0"])

    def test_module_all_exports(self):
        from core.tensor_parallel import __all__
        assert "apply_tensor_parallel" in __all__
        assert "TPModel" in __all__

    def test_paged_attention_cuda_all_exports(self):
        from core.paged_attention_cuda import __all__
        assert "paged_attention_decode" in __all__


# ================================================================
# Arch detection helper
# ================================================================

class TestCUDAArchDetection:
    """Test the CUDA architecture auto-detection."""

    def test_detect_function_exists(self):
        from core.paged_attention_cuda import _detect_and_set_cuda_arch
        assert callable(_detect_and_set_cuda_arch)

    @skip_no_cuda
    def test_detect_sets_env(self):
        """After detection, TORCH_CUDA_ARCH_LIST should be set."""
        import os
        from core.paged_attention_cuda import _detect_and_set_cuda_arch
        # Only test if not already set
        saved = os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        try:
            _detect_and_set_cuda_arch()
            val = os.environ.get("TORCH_CUDA_ARCH_LIST")
            assert val is not None
            # Should contain digits and dots
            assert "." in val
        finally:
            if saved is not None:
                os.environ["TORCH_CUDA_ARCH_LIST"] = saved
            else:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
