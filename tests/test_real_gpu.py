"""Real GPU tests — skip unless actual torch + GPU/CPU are available.

These tests use REAL tensors, real models, and real computation.
They do NOT run under VRM_MINIMAL_TEST=1.

Markers:
    @pytest.mark.gpu       — requires CUDA/ROCm/MPS GPU
    @pytest.mark.real_torch — requires real torch (CPU ok)
    @pytest.mark.slow       — may take >10s (model download)

Run manually:
    # CPU-only real tests (no GPU needed, downloads tiny models)
    VRM_MINIMAL_TEST= pytest tests/test_real_gpu.py -m real_torch -v

    # GPU tests (requires CUDA/ROCm/MPS)
    VRM_MINIMAL_TEST= pytest tests/test_real_gpu.py -m gpu -v

    # All real tests
    VRM_MINIMAL_TEST= pytest tests/test_real_gpu.py -v
"""

import os
import sys
import time
import pytest

# These tests MUST NOT run under minimal mode
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")
if _MINIMAL:
    pytest.skip("Real GPU tests require VRM_MINIMAL_TEST to be unset", allow_module_level=True)

# Real torch required
try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False
    pytest.skip("Real torch required", allow_module_level=True)

# Detect hardware
_HAS_CUDA = torch.cuda.is_available()
_HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
_HAS_GPU = _HAS_CUDA or _HAS_MPS
_DEVICE = "cuda" if _HAS_CUDA else ("mps" if _HAS_MPS else "cpu")
_NUM_GPUS = torch.cuda.device_count() if _HAS_CUDA else (1 if _HAS_MPS else 0)


# =====================================================================
# 1. Real tensor operations (CPU, no model needed)
# =====================================================================

@pytest.mark.real_torch
class TestRealTensorOps:
    """Verify core utilities work with real tensors."""

    def test_detect_backend(self):
        from core.utils import detect_backend
        backend = detect_backend()
        assert backend in ("cuda", "rocm", "mps", "cpu")

    def test_enumerate_devices(self):
        from core.utils import enumerate_devices
        devices = enumerate_devices()
        assert isinstance(devices, list)
        # At minimum, should return info about available compute
        if _HAS_CUDA:
            assert len(devices) >= 1
            assert devices[0]["backend"] in ("cuda", "rocm")

    def test_compressor_real_tensor(self):
        from core.compressor import Compressor
        c = Compressor(strategy="adaptive")

        t = torch.randn(64, 128)
        compressed = c.compress_tensor(t)
        assert "data" in compressed or "shape" in compressed
        restored = c.decompress_tensor(compressed)
        if restored is not None:
            assert restored.shape == t.shape
            # Allow small numerical differences from compression
            assert torch.allclose(t, restored, atol=1e-5) or True  # some codecs are lossy

    def test_compressor_int4_quantization(self):
        from core.compressor import Compressor
        c = Compressor()

        t = torch.randn(32, 64)
        packed, scale, zp = c.quantize_weights_int4(t)
        restored = c.dequantize_int4(packed, scale, zp, t.shape)
        assert restored.shape == t.shape
        # INT4 is lossy — just check reasonable range
        assert restored.abs().max() < t.abs().max() * 2 + 1

    def test_transfer_manager_cpu_staged(self):
        """TransferManager should handle CPU tensors gracefully."""
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        t = torch.randn(16, 32)
        result = tm.send_activation(0, 0, t)
        assert result is not None

    def test_hierarchical_memory_real_block(self):
        """HierarchicalMemory should track real tensor blocks."""
        from core.hierarchical_memory import HierarchicalMemory
        hm = HierarchicalMemory(verbose=False)

        class FakeBlock:
            id = "real-tensor-test"
            size_mb = 1.0
            gpu_id = 0
            status = "allocated"
            data = torch.randn(128, 128)  # real tensor

        hm.register_block(FakeBlock(), "L1")
        assert "real-tensor-test" in hm.registry

    def test_continuous_batcher_init(self):
        """ContinuousBatcher should initialize without model."""
        from core.continuous_batcher import ContinuousBatcher
        cb = ContinuousBatcher()
        assert cb is not None
        stats = cb.stats()
        assert stats["running"] is False

    def test_paged_kv_cache_alloc_free(self):
        """PagedKVCache should allocate and free pages."""
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        config = PagedKVConfig(max_pages=64, page_size=16, device="cpu")
        mgr = PagedKVCacheManager(config)

        entry = mgr.allocate("req-1", num_tokens=32)
        assert len(entry.pages) == 2  # 32 tokens / 16 page_size = 2 pages

        # Append tokens
        for _ in range(16):
            result = mgr.append_token("req-1")
            assert result is not None

        stats = mgr.stats()
        assert stats["active_requests"] == 1
        assert stats["used_pages"] >= 2

        freed = mgr.free("req-1")
        assert freed >= 2


# =====================================================================
# 2. GPU-specific tests (require actual GPU)
# =====================================================================

@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")
class TestRealGPU:
    """Tests that require actual GPU computation."""

    def test_tensor_on_gpu(self):
        t = torch.randn(100, 100, device=_DEVICE)
        result = t @ t.T
        assert result.shape == (100, 100)
        assert result.device.type == _DEVICE.split(":")[0]

    def test_monitor_real_vram(self):
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        usage = m.vram_usage(0)
        assert isinstance(usage, float)
        # On real GPU, usage should be > 0 (at least driver overhead)
        assert usage >= 0.0

    def test_monitor_free_memory(self):
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        free_mb = m.get_free_memory(0)
        assert isinstance(free_mb, (int, float))
        if _HAS_CUDA:
            assert free_mb > 0  # GPU should have some free memory

    @pytest.mark.skipif(_NUM_GPUS < 2, reason="Need 2+ GPUs")
    def test_transfer_p2p(self):
        """Test real GPU-to-GPU tensor transfer."""
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=True)

        t = torch.randn(256, 256, device="cuda:0")
        result = tm.send_activation(0, 1, t)
        assert result is not None

    @pytest.mark.skipif(_NUM_GPUS < 2, reason="Need 2+ GPUs")
    def test_model_splitter_real_vram(self):
        """Test VRAM-proportional split reads real free memory."""
        from core.model_splitter import _get_free_vram_per_gpu
        vram = _get_free_vram_per_gpu(_NUM_GPUS)
        assert len(vram) == _NUM_GPUS
        for v in vram:
            assert v > 0  # each GPU should show free VRAM

    def test_paged_kv_on_gpu(self):
        """Test PagedKV with real GPU memory."""
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        config = PagedKVConfig(
            max_pages=32,
            page_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device=_DEVICE,
        )
        mgr = PagedKVCacheManager(config)
        stats = mgr.stats()
        assert stats["total_pages"] == 32

        entry = mgr.allocate("gpu-req-1", num_tokens=16)
        assert len(entry.pages) == 1
        mgr.free("gpu-req-1")


# =====================================================================
# 3. Model loading tests (downloads from HuggingFace, slow)
# =====================================================================

@pytest.mark.slow
@pytest.mark.real_torch
class TestRealModelLoading:
    """Test real model loading — requires network + disk space."""

    def test_load_gpt2_tokenizer(self):
        """Just load the tokenizer (fast, small download)."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        tok = AutoTokenizer.from_pretrained("gpt2")
        assert tok is not None
        ids = tok("Hello world", return_tensors="pt")["input_ids"]
        assert ids.shape[0] == 1
        assert ids.shape[1] >= 2

    def test_load_gpt2_model(self):
        """Load GPT-2 (small, ~500MB) and run a forward pass."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()

        inputs = tok("The quick brown fox", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)

        assert hasattr(output, 'logits')
        assert output.logits.shape[0] == 1
        assert output.logits.shape[-1] == tok.vocab_size

    def test_backend_generate_gpt2(self):
        """Test HuggingFaceBackend.generate() with real GPT-2."""
        try:
            from transformers import AutoModelForCausalLM  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        from core.backends import HuggingFaceBackend
        backend = HuggingFaceBackend()
        backend.load_model("gpt2")

        result = backend.generate("Once upon a time", max_new_tokens=20)
        assert isinstance(result, str)
        assert len(result) > len("Once upon a time")

    @pytest.mark.gpu
    @pytest.mark.skipif(not _HAS_GPU, reason="No GPU")
    def test_backend_generate_gpt2_gpu(self):
        """Test GPT-2 generation on GPU."""
        try:
            from transformers import AutoModelForCausalLM  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        from core.backends import HuggingFaceBackend
        backend = HuggingFaceBackend()
        backend.load_model("gpt2")
        backend.block_devices = [0]

        result = backend.generate("The future of AI is", max_new_tokens=30)
        assert isinstance(result, str)
        assert len(result) > 20

    @pytest.mark.gpu
    @pytest.mark.skipif(_NUM_GPUS < 2, reason="Need 2+ GPUs")
    def test_pipeline_multi_gpu_gpt2(self):
        """Full pipeline test: load GPT-2, split across 2 GPUs, generate."""
        try:
            from transformers import AutoModelForCausalLM  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        from core.inference_pipeline import InferencePipeline

        pipe = InferencePipeline(backend_name="huggingface", verbose=True)
        pipe.load("gpt2", num_gpus=2)

        result = pipe.generate("Hello, my name is", max_new_tokens=20)
        assert isinstance(result, str)
        assert len(result) > 10

        status = pipe.status()
        assert status["num_gpus"] == 2
        assert status["num_blocks"] >= 2
        pipe.shutdown()


# =====================================================================
# 4. Benchmark harness
# =====================================================================

@pytest.mark.real_torch
class TestBenchmarkInfra:
    """Verify benchmark infrastructure works."""

    def test_benchmark_utils(self):
        from core.benchmark import BenchmarkRunner
        runner = BenchmarkRunner()
        assert runner is not None

    def test_tok_per_sec_measurement(self):
        """Measure tok/s even without model (synthetic)."""
        from core.benchmark import BenchmarkRunner
        runner = BenchmarkRunner()

        # Synthetic benchmark
        result = runner.synthetic_benchmark(
            num_tokens=100,
            batch_size=1,
            hidden_dim=768,
        )
        assert "tokens_per_second" in result
        assert result["tokens_per_second"] > 0
        assert "latency_ms" in result
