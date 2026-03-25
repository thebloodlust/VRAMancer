"""Tests for DirectFP4 bypass (core/nvfp4_direct.py).

Tests cover:
  1. Module import and class availability (minimal mode)
  2. DirectFP4Linear construction and buffer registration
  3. from_nvfp4_tensor() extraction (mocked NVFP4Tensor)
  4. replace_with_direct_fp4() model surgery
  5. DirectFP4 bypass integration in backends.py
  6. Forward pass correctness (requires real torch + torchao + Blackwell GPU)
"""
import os
import sys
import types
import inspect
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Check for real torch availability (not conftest SimpleNamespace mock)
try:
    import torch
    HAS_TORCH = hasattr(torch.nn, 'Module') and torch.nn.Module is not object
except (ImportError, AttributeError):
    HAS_TORCH = False

# Check for real torchao NVFP4 availability
try:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    HAS_TORCHAO = True
except Exception:
    HAS_TORCHAO = False
    NVFP4Tensor = None

# Check for Blackwell GPU (CC >= 10.0)
HAS_BLACKWELL = False
if HAS_TORCH and torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(i)[0] >= 10:
            HAS_BLACKWELL = True
            break


class TestDirectFP4Import:
    """Test that core.nvfp4_direct imports cleanly."""

    def test_module_imports(self):
        """core.nvfp4_direct should import without error."""
        import core.nvfp4_direct as mod
        assert hasattr(mod, 'DirectFP4Linear')
        assert hasattr(mod, 'replace_with_direct_fp4')

    def test_has_torch_flag(self):
        """Module exposes HAS_TORCH and HAS_TORCHAO flags."""
        import core.nvfp4_direct as mod
        assert hasattr(mod, 'HAS_TORCH')
        assert hasattr(mod, 'HAS_TORCHAO')

    def test_class_has_expected_methods(self):
        """DirectFP4Linear has the expected API."""
        from core.nvfp4_direct import DirectFP4Linear
        assert hasattr(DirectFP4Linear, 'from_nvfp4_tensor')
        assert hasattr(DirectFP4Linear, 'from_linear')
        assert hasattr(DirectFP4Linear, 'forward')
        assert hasattr(DirectFP4Linear, '_init_views')
        assert hasattr(DirectFP4Linear, '_quantize_activation_triton')
        assert hasattr(DirectFP4Linear, '_quantize_activation_python')


class TestDirectFP4Construction:
    """Test DirectFP4Linear construction (requires real torch)."""

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires real torch")
    def test_basic_construction(self):
        """DirectFP4Linear can be instantiated with dimensions."""
        from core.nvfp4_direct import DirectFP4Linear
        mod = DirectFP4Linear(in_features=128, out_features=64)
        assert mod.in_features == 128
        assert mod.out_features == 64
        assert mod.w_qdata is None
        assert mod.bias is None

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires real torch")
    def test_construction_with_bias(self):
        """DirectFP4Linear accepts optional bias."""
        from core.nvfp4_direct import DirectFP4Linear
        bias = torch.randn(64)
        mod = DirectFP4Linear(in_features=128, out_features=64, bias_data=bias)
        assert mod.bias is not None
        assert mod.bias.shape == (64,)

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires real torch")
    def test_buffers_registered(self):
        """w_qdata, w_scale, w_per_tensor_scale are registered buffers."""
        from core.nvfp4_direct import DirectFP4Linear
        mod = DirectFP4Linear(in_features=128, out_features=64)
        buffer_names = [n for n, _ in mod.named_buffers()]
        assert 'w_qdata' in buffer_names
        assert 'w_scale' in buffer_names
        assert 'w_per_tensor_scale' in buffer_names


class TestFromNVFP4Tensor:
    """Test from_nvfp4_tensor() extraction."""

    @pytest.mark.skipif(not (HAS_TORCH and HAS_TORCHAO),
                        reason="Requires torch + torchao")
    def test_from_nvfp4_tensor_extracts_data(self):
        """from_nvfp4_tensor creates DirectFP4Linear from NVFP4Tensor weight."""
        from core.nvfp4_direct import DirectFP4Linear

        # Create a mock NVFP4Tensor with the expected attributes
        mock_weight = MagicMock(spec=NVFP4Tensor)
        mock_weight.shape = (64, 128)
        mock_weight.orig_dtype = torch.bfloat16
        mock_weight.use_triton_kernel = True
        mock_weight.is_swizzled_scales = True
        mock_weight.qdata = torch.randint(0, 255, (64, 64), dtype=torch.uint8)
        mock_weight.scale = torch.randn(64, 8, dtype=torch.float32)
        mock_weight.per_tensor_scale = torch.tensor(1.0)

        mock_linear = MagicMock(spec=torch.nn.Linear)
        mock_linear.bias = None

        direct = DirectFP4Linear.from_nvfp4_tensor(mock_linear, mock_weight)
        assert direct.in_features == 128
        assert direct.out_features == 64
        assert direct.w_qdata is not None
        assert direct.w_scale is not None
        assert direct.w_per_tensor_scale is not None


class TestReplaceWithDirectFP4:
    """Test replace_with_direct_fp4() model surgery."""

    @pytest.mark.skipif(not HAS_TORCHAO, reason="Requires torchao NVFP4")
    def test_replace_returns_zero_for_normal_model(self):
        """No NVFP4Tensor weights → replace_with_direct_fp4 returns 0."""
        from core.nvfp4_direct import replace_with_direct_fp4

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        )
        count = replace_with_direct_fp4(model, verbose=False)
        assert count == 0

    def test_replace_returns_zero_without_torchao(self):
        """Without torchao, replace_with_direct_fp4 returns 0 safely."""
        from core.nvfp4_direct import replace_with_direct_fp4
        if not HAS_TORCHAO:
            model = MagicMock()
            model.named_modules.return_value = []
            count = replace_with_direct_fp4(model, verbose=False)
            assert count == 0


class TestBackendsIntegration:
    """Verify DirectFP4 bypass is integrated in backends.py."""

    def test_apply_nvfp4_calls_direct_fp4(self):
        """_apply_nvfp4_quantization source references replace_with_direct_fp4."""
        from core.backends import HuggingFaceBackend
        source = inspect.getsource(HuggingFaceBackend._apply_nvfp4_quantization)
        assert "replace_with_direct_fp4" in source
        assert "DirectFP4 bypass" in source

    def test_direct_fp4_has_fallback(self):
        """DirectFP4 integration has a try/except fallback."""
        from core.backends import HuggingFaceBackend
        source = inspect.getsource(HuggingFaceBackend._apply_nvfp4_quantization)
        assert "except" in source
        assert "unavailable" in source.lower() or "warning" in source.lower()


class TestFromLinear:
    """Test from_linear() quantization path."""

    @pytest.mark.skipif(not (HAS_TORCH and HAS_TORCHAO and HAS_BLACKWELL),
                        reason="Requires torch + torchao + Blackwell GPU")
    @pytest.mark.heavy
    def test_from_linear_roundtrip(self):
        """from_linear quantizes a Linear layer and produces valid DirectFP4Linear."""
        from core.nvfp4_direct import DirectFP4Linear

        # Find Blackwell GPU
        bw_gpu = None
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_capability(i)[0] >= 10:
                bw_gpu = i
                break

        device = torch.device(f"cuda:{bw_gpu}")
        torch.cuda.set_device(device)

        linear = torch.nn.Linear(256, 128, bias=True).to(device).to(torch.bfloat16)
        direct = DirectFP4Linear.from_linear(linear, device=device)

        assert direct.in_features == 256
        assert direct.out_features == 128
        assert direct.w_qdata is not None
        assert direct.bias is not None

        # Forward pass should work
        x = torch.randn(2, 256, dtype=torch.bfloat16, device=device)
        out = direct.to(device)(x)
        assert out.shape == (2, 128)
        assert not torch.isnan(out).any()


class TestForwardCorrectness:
    """Test forward pass matches torchao output (requires Blackwell GPU)."""

    @pytest.mark.skipif(not (HAS_TORCH and HAS_TORCHAO and HAS_BLACKWELL),
                        reason="Requires torch + torchao + Blackwell GPU")
    @pytest.mark.heavy
    def test_forward_matches_torchao(self):
        """DirectFP4Linear output matches NVFP4Tensor output exactly."""
        from torchao.quantization import quantize_
        from torchao.prototype.mx_formats import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        from core.nvfp4_direct import DirectFP4Linear

        # Find Blackwell GPU
        bw_gpu = None
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_capability(i)[0] >= 10:
                bw_gpu = i
                break

        device = torch.device(f"cuda:{bw_gpu}")
        torch.cuda.set_device(device)

        # Create small model, quantize with torchao
        model = torch.nn.Linear(256, 128, bias=False).to(torch.bfloat16).to(device)
        quantize_(model, NVFP4DynamicActivationNVFP4WeightConfig())

        # Get torchao output
        x = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            ref_out = model(x)

        # Create DirectFP4Linear from the quantized weight
        direct = DirectFP4Linear.from_nvfp4_tensor(model, model.weight)
        direct = direct.to(device)

        with torch.no_grad():
            direct_out = direct(x)

        # Should be exact match (same cuBLAS call, same data)
        max_err = (ref_out - direct_out).abs().max().item()
        assert max_err == 0.0, f"Max error: {max_err}"
