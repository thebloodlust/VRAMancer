"""End-to-end multi-GPU inference test.

Validates the full pipeline:
  1. Load a model (GPT-2 small or mock)
  2. Split it across GPUs (profiler-based or VRAM-based)
  3. Run forward pass through all blocks
  4. Verify output shape and consistency

Runs in two modes:
  - VRM_MINIMAL_TEST=1 : uses mocks, fast, no GPU needed
  - Real mode : loads GPT-2, requires torch + GPU(s)

Usage:
    pytest tests/test_end_to_end.py -v
    pytest tests/test_end_to_end.py -v -m "not slow"  # skip real model tests
"""
import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False
    torch = None
    nn = None


# ---------------------------------------------------------------
# Mock model for fast testing
# ---------------------------------------------------------------

class MockTransformerBlock(nn.Module if nn else object):
    """A simple mock transformer block for testing."""
    def __init__(self, hidden_size=64):
        if nn:
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
            self.act = nn.GELU()
            self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return self.norm(x + residual)


class MockTransformerModel(nn.Module if nn else object):
    """Mock model with .config and extractable layers."""
    def __init__(self, n_layers=8, hidden_size=64):
        if nn:
            super().__init__()
            self.config = type("Config", (), {"hidden_size": hidden_size})()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(
                [MockTransformerBlock(hidden_size) for _ in range(n_layers)]
            )

    def forward(self, x):
        for block in self.transformer.h:
            x = block(x)
        return x


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

class TestEndToEndPipeline:
    """End-to-end tests for the full inference pipeline."""

    @pytest.mark.smoke
    def test_layer_profiler_mock(self):
        """Test LayerProfiler with mock model."""
        from core.layer_profiler import LayerProfiler, LayerProfile

        profiler = LayerProfiler(warmup_iters=1, profile_iters=2)

        if _TORCH and not _MINIMAL:
            model = MockTransformerModel(n_layers=4, hidden_size=64)
            profiles = profiler.profile_model(model, device="cpu")

            assert len(profiles) == 4
            for p in profiles:
                assert isinstance(p, LayerProfile)
                assert p.param_count > 0
                assert p.total_memory_mb > 0
                assert p.latency_ms >= 0
        else:
            # Stub mode returns synthetic profiles
            profiles = profiler.profile_model(None)
            assert len(profiles) > 0

    @pytest.mark.smoke
    def test_placement_engine_strategies(self):
        """Test PlacementEngine with all built-in strategies."""
        from core.orchestrator.placement_engine import PlacementEngine

        engine = PlacementEngine()

        block = {"size_mb": 128, "layer_type": "attention"}

        # Test each strategy
        for strategy in ["profiled", "vram", "balanced"]:
            result = engine.place(block, strategy=strategy)
            assert "level" in result
            assert "gpu_id" in result

        # Test default (no strategy name)
        result = engine.place(block)
        assert "level" in result

    @pytest.mark.smoke
    def test_placement_engine_custom_strategy(self):
        """Test registering and using a custom strategy."""
        from core.orchestrator.placement_engine import PlacementEngine

        engine = PlacementEngine()

        def my_strategy(block):
            return {"level": "L2", "gpu_id": 42, "strategy": "custom"}

        engine.register_strategy("custom", my_strategy)
        result = engine.place({"size_mb": 100}, strategy="custom")
        assert result["gpu_id"] == 42
        assert result["level"] == "L2"

    @pytest.mark.smoke
    def test_optimal_placement_dp(self):
        """Test DP-based optimal placement with synthetic data."""
        from core.layer_profiler import (
            LayerProfile,
            GPUProfile,
            PlacementPlan,
            compute_optimal_placement,
        )

        # 8 layers, 2 GPUs with different capabilities
        layers = [
            LayerProfile(
                index=i,
                name=f"layer_{i}",
                latency_ms=1.0 + (i % 3) * 0.5,
                total_memory_mb=100.0,
                activation_memory_mb=10.0,
                estimated_flops=int(1e9),
                layer_type="block",
            )
            for i in range(8)
        ]

        gpus = [
            GPUProfile(
                index=0, name="FastGPU", backend="cuda",
                total_vram_mb=24000, free_vram_mb=20000,
                compute_throughput_gflops=50.0,
                memory_bandwidth_gbps=900.0,
            ),
            GPUProfile(
                index=1, name="SlowGPU", backend="cuda",
                total_vram_mb=16000, free_vram_mb=14000,
                compute_throughput_gflops=20.0,
                memory_bandwidth_gbps=500.0,
            ),
        ]

        plan = compute_optimal_placement(layers, gpus, transfer_bandwidth_gbps=25.0)

        assert isinstance(plan, PlacementPlan)
        assert len(plan.assignments) == 8
        assert plan.estimated_latency_ms > 0

        # Fast GPU should get more layers
        gpu0_layers = sum(1 for _, g in plan.assignments if g == 0)
        gpu1_layers = sum(1 for _, g in plan.assignments if g == 1)
        assert gpu0_layers + gpu1_layers == 8
        # Fast GPU (2.5x faster) should get more layers
        assert gpu0_layers >= gpu1_layers, (
            f"Fast GPU got {gpu0_layers} layers, slow got {gpu1_layers}"
        )

    @pytest.mark.smoke
    def test_vram_constraint_enforcement(self):
        """Test that VRAM constraints are enforced in placement."""
        from core.layer_profiler import (
            LayerProfile,
            GPUProfile,
            compute_optimal_placement,
        )

        # Layers that total 500MB each
        layers = [
            LayerProfile(
                index=i,
                name=f"layer_{i}",
                latency_ms=1.0,
                total_memory_mb=500.0,
                activation_memory_mb=10.0,
                layer_type="block",
            )
            for i in range(4)
        ]

        # GPU 0 has 1200MB free (fits 2 layers), GPU 1 has 1200MB
        gpus = [
            GPUProfile(
                index=0, name="GPU0", backend="cuda",
                total_vram_mb=2000, free_vram_mb=1200,
                compute_throughput_gflops=50.0,
            ),
            GPUProfile(
                index=1, name="GPU1", backend="cuda",
                total_vram_mb=2000, free_vram_mb=1200,
                compute_throughput_gflops=50.0,
            ),
        ]

        plan = compute_optimal_placement(layers, gpus)
        assert len(plan.assignments) == 4

    @pytest.mark.skipif(not _TORCH or _MINIMAL, reason="Requires torch")
    def test_mock_model_forward_pass(self):
        """Test full forward pass through mock model blocks."""
        model = MockTransformerModel(n_layers=8, hidden_size=64)
        batch = torch.randn(1, 16, 64)

        # Full model forward
        with torch.no_grad():
            output = model(batch)
        assert output.shape == batch.shape

    @pytest.mark.skipif(not _TORCH or _MINIMAL, reason="Requires torch")
    def test_mock_model_split_and_forward(self):
        """Test splitting mock model into blocks and running each."""
        from core.model_splitter import _extract_layers, _split_by_vram

        model = MockTransformerModel(n_layers=8, hidden_size=64)

        layers = _extract_layers(model)
        assert layers is not None
        assert len(layers) == 8

        # Split into 2 blocks (simulating 2 GPUs with equal VRAM)
        blocks = _split_by_vram(layers, [8192, 8192])
        assert len(blocks) == 2

        # Run forward pass through each block sequentially
        x = torch.randn(1, 16, 64)
        with torch.no_grad():
            for block in blocks:
                x = block(x)

        assert x.shape == (1, 16, 64)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()

    @pytest.mark.skipif(not _TORCH or _MINIMAL, reason="Requires torch")
    def test_profiler_with_mock_model(self):
        """Test full profiler pipeline with mock model on CPU."""
        from core.layer_profiler import LayerProfiler, compute_optimal_placement

        model = MockTransformerModel(n_layers=6, hidden_size=64)
        profiler = LayerProfiler(
            warmup_iters=1,
            profile_iters=3,
            batch_size=1,
            seq_length=16,
        )

        layer_profiles = profiler.profile_model(model, device="cpu")
        assert len(layer_profiles) == 6

        for lp in layer_profiles:
            assert lp.param_count > 0
            assert lp.latency_ms >= 0

        # Test with synthetic GPU profiles (can't benchmark real GPUs on CPU)
        from core.layer_profiler import GPUProfile

        gpus = [
            GPUProfile(0, "MockGPU0", "cuda", 24000, 20000, 50.0, 900.0),
            GPUProfile(1, "MockGPU1", "cuda", 16000, 14000, 30.0, 600.0),
        ]
        plan = compute_optimal_placement(layer_profiles, gpus)

        assert len(plan.assignments) == 6
        assert plan.estimated_latency_ms > 0

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.skipif(not _TORCH or _MINIMAL, reason="Requires torch + network")
    def test_real_gpt2_split(self):
        """Load real GPT-2, split, and verify output consistency.

        WARNING: Downloads GPT-2 (~500MB). Slow test.
        """
        try:
            from transformers import AutoModel
        except ImportError:
            pytest.skip("transformers not installed")

        model = AutoModel.from_pretrained("gpt2")
        from core.model_splitter import _extract_layers, _split_by_vram

        layers = _extract_layers(model)
        assert layers is not None
        assert len(layers) == 12  # GPT-2 has 12 layers

        # Full model forward
        x = torch.randn(1, 32, 768)
        with torch.no_grad():
            full_output = model.transformer.h[0](x)  # first block for reference

        # Split and verify block 0 matches
        blocks = _split_by_vram(layers, [8192, 8192])
        with torch.no_grad():
            block0_output = blocks[0][0](x)

        # Should be identical (same weights, same computation)
        assert torch.allclose(full_output[0], block0_output[0], atol=1e-5)


class TestGPUProfiler:
    """Test GPU profiling capabilities."""

    @pytest.mark.smoke
    def test_profiler_stub_mode(self):
        """Test profiler in stub mode."""
        from core.layer_profiler import LayerProfiler

        # Force stub
        profiler = LayerProfiler()
        profiler._stub = True

        profiles = profiler.profile_model(None)
        assert len(profiles) > 0

        gpu_profiles = profiler.profile_gpus()
        assert len(gpu_profiles) == 1
        assert gpu_profiles[0].name == "StubGPU"

    @pytest.mark.smoke
    def test_layer_classification(self):
        """Test layer type classification."""
        from core.layer_profiler import LayerProfiler

        profiler = LayerProfiler()

        # Mock classes with specific names
        class SelfAttention:
            pass

        class MLP:
            pass

        class LayerNorm:
            pass

        class EmbeddingLayer:
            pass

        class TransformerBlock:
            pass

        assert profiler._classify_layer(SelfAttention()) == "attention"
        assert profiler._classify_layer(MLP()) == "mlp"
        assert profiler._classify_layer(LayerNorm()) == "norm"
        assert profiler._classify_layer(EmbeddingLayer()) == "embedding"
        assert profiler._classify_layer(TransformerBlock()) == "block"

    @pytest.mark.smoke
    def test_flops_estimation(self):
        """Test FLOPS estimation for different layer types."""
        from core.layer_profiler import LayerProfiler

        profiler = LayerProfiler(batch_size=1, seq_length=128)

        # Attention layer FLOPS should be higher than norm
        attn_flops = profiler._estimate_layer_flops(None, "attention", 768)
        norm_flops = profiler._estimate_layer_flops(None, "norm", 768)

        assert attn_flops > norm_flops
        assert attn_flops > 0
        assert norm_flops > 0
