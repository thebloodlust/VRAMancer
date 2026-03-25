"""Tests for NVFP4 Blackwell quantization support in VRAMancer.

Covers:
  1. _get_quantization_mode() returns 'nvfp4' on Blackwell GPUs
  2. _get_quantization_mode() falls back to 'nf4' on non-Blackwell
  3. _has_blackwell_gpu() detection
  4. _nvfp4_filter_fn() excludes lm_head
  5. _apply_nvfp4_quantization code is present in backends.py
  6. NVFP4 load_model flow (post-load quantization path)
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


class TestNVFP4QuantizationMode:
    """Test _get_quantization_mode() NVFP4 routing."""

    def _make_backend(self):
        from core.backends import HuggingFaceBackend
        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend.log = MagicMock()
        return backend

    @patch.dict(os.environ, {"VRM_QUANTIZATION": "nvfp4"})
    def test_nvfp4_returns_nvfp4_on_blackwell(self):
        """When a Blackwell GPU is present and torchao available, return 'nvfp4'."""
        backend = self._make_backend()
        with patch.object(type(backend), '_has_blackwell_gpu', return_value=True):
            # Mock torchao import
            nvfp4_mod = types.ModuleType("torchao.prototype.mx_formats.nvfp4_tensor")
            nvfp4_mod.NVFP4Tensor = MagicMock
            with patch.dict(sys.modules, {
                "torchao.prototype.mx_formats.nvfp4_tensor": nvfp4_mod,
            }):
                # Also need torch + bitsandbytes for fallback check
                torch_mock = MagicMock()
                torch_mock.cuda.is_available.return_value = True
                with patch.dict(sys.modules, {"torch": torch_mock}):
                    mode = backend._get_quantization_mode()
        assert mode == "nvfp4"

    @patch.dict(os.environ, {"VRM_QUANTIZATION": "nvfp4"})
    def test_nvfp4_falls_back_to_nf4_on_non_blackwell(self):
        """When no Blackwell GPU, nvfp4 falls back to 'nf4' (if BnB available)."""
        backend = self._make_backend()
        with patch.object(type(backend), '_has_blackwell_gpu', return_value=False):
            mode = backend._get_quantization_mode()
        # Falls back to nf4 or '' depending on bitsandbytes availability
        assert mode in ("nf4", "")

    @patch.dict(os.environ, {"VRM_QUANTIZATION": "nf4"})
    def test_nf4_mode_unchanged(self):
        """VRM_QUANTIZATION=nf4 still returns 'nf4' (no NVFP4 routing)."""
        backend = self._make_backend()
        mode = backend._get_quantization_mode()
        # Returns 'nf4' or '' depending on BnB/torch availability
        assert mode in ("nf4", "")

    @patch.dict(os.environ, {"VRM_QUANTIZATION": ""})
    def test_empty_mode(self):
        """No VRM_QUANTIZATION returns empty string."""
        backend = self._make_backend()
        assert backend._get_quantization_mode() == ""


class TestBlackwellDetection:
    """Test _has_blackwell_gpu() static method."""

    def test_has_blackwell_with_cc_12(self):
        """CC 12.0 (RTX 5070 Ti) is Blackwell."""
        from core.backends import HuggingFaceBackend
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 2
        torch_mock.cuda.get_device_capability.side_effect = [
            (8, 6),   # GPU 0: Ampere (3090)
            (12, 0),  # GPU 1: Blackwell (5070 Ti)
        ]
        import core.backends as _bmod
        with patch.object(_bmod, '_HAS_TORCH', True), \
             patch.object(_bmod, '_torch', torch_mock):
            assert HuggingFaceBackend._has_blackwell_gpu() is True

    def test_no_blackwell_with_ampere_only(self):
        """CC 8.6 (RTX 3090) is not Blackwell."""
        from core.backends import HuggingFaceBackend
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 1
        torch_mock.cuda.get_device_capability.return_value = (8, 6)
        import core.backends as _bmod
        with patch.object(_bmod, '_HAS_TORCH', True), \
             patch.object(_bmod, '_torch', torch_mock):
            assert HuggingFaceBackend._has_blackwell_gpu() is False

    def test_no_cuda_returns_false(self):
        """No CUDA available -> False."""
        from core.backends import HuggingFaceBackend
        import core.backends as _bmod
        with patch.object(_bmod, '_HAS_TORCH', False):
            assert HuggingFaceBackend._has_blackwell_gpu() is False


class TestNVFP4FilterFn:
    """Test _nvfp4_filter_fn() for lm_head exclusion."""

    def test_excludes_lm_head(self):
        """lm_head should be excluded from NVFP4 quantization."""
        from core.backends import HuggingFaceBackend
        import core.backends as _bmod
        if not _bmod._HAS_TORCH:
            # Create mock Linear
            linear = MagicMock()
            linear.__class__.__name__ = "Linear"
            # In minimal test, torch is mocked so isinstance won't work
            pytest.skip("Requires real torch for isinstance check")
        else:
            linear = _bmod._torch.nn.Linear(10, 10)
            assert HuggingFaceBackend._nvfp4_filter_fn(linear, "model.lm_head") is False
            assert HuggingFaceBackend._nvfp4_filter_fn(linear, "model.embed_tokens") is False
            assert HuggingFaceBackend._nvfp4_filter_fn(linear, "model.layers.0.self_attn.q_proj") is True

    def test_excludes_non_linear(self):
        """Non-Linear modules should be excluded."""
        from core.backends import HuggingFaceBackend
        import core.backends as _bmod
        if not _bmod._HAS_TORCH:
            pytest.skip("Requires real torch")
        else:
            norm = _bmod._torch.nn.LayerNorm(10)
            assert HuggingFaceBackend._nvfp4_filter_fn(norm, "model.layers.0.norm") is False


class TestNVFP4CodePresence:
    """Verify NVFP4 code is properly integrated in backends.py."""

    def test_nvfp4_quantization_method_exists(self):
        """_apply_nvfp4_quantization method exists on HuggingFaceBackend."""
        from core.backends import HuggingFaceBackend
        assert hasattr(HuggingFaceBackend, '_apply_nvfp4_quantization')
        assert callable(getattr(HuggingFaceBackend, '_apply_nvfp4_quantization'))

    def test_nvfp4_filter_fn_exists(self):
        """_nvfp4_filter_fn static method exists."""
        from core.backends import HuggingFaceBackend
        assert hasattr(HuggingFaceBackend, '_nvfp4_filter_fn')

    def test_nvfp4_code_references(self):
        """Key NVFP4 code patterns exist in HuggingFaceBackend source."""
        from core.backends import HuggingFaceBackend
        source = inspect.getsource(HuggingFaceBackend)
        assert "NVFP4DynamicActivationNVFP4WeightConfig" in source
        assert "quantize_" in source
        assert "float4_e2m1fn" in source or "nvfp4" in source.lower()
        assert "_nvfp4_filter_fn" in source
        assert "cublas" in source.lower() or "scaled_mm" in source

    def test_load_model_has_nvfp4_branch(self):
        """load_model() contains the NVFP4 post-load quantization path."""
        from core.backends import HuggingFaceBackend
        source = inspect.getsource(HuggingFaceBackend.load_model)
        assert 'quant_mode == "nvfp4"' in source
        assert "_apply_nvfp4_quantization" in source
        assert 'device_map' in source

    def test_get_quantization_mode_handles_nvfp4(self):
        """_get_quantization_mode() distinguishes nvfp4 from nf4."""
        from core.backends import HuggingFaceBackend
        source = inspect.getsource(HuggingFaceBackend._get_quantization_mode)
        assert '"nvfp4"' in source
        assert "_has_blackwell_gpu" in source
        assert "torchao" in source


class TestNVFP4ShouldUseCompat:
    """Test _should_use_nvfp4 backward compat."""

    def _make_backend(self):
        from core.backends import HuggingFaceBackend
        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend.log = MagicMock()
        return backend

    @patch.dict(os.environ, {"VRM_QUANTIZATION": "nvfp4"})
    def test_should_use_nvfp4_returns_true_for_nvfp4(self):
        """_should_use_nvfp4() returns True when mode is 'nvfp4'."""
        backend = self._make_backend()
        with patch.object(type(backend), '_get_quantization_mode', return_value='nvfp4'):
            assert backend._should_use_nvfp4() is True

    @patch.dict(os.environ, {"VRM_QUANTIZATION": "nf4"})
    def test_should_use_nvfp4_returns_true_for_nf4(self):
        """_should_use_nvfp4() returns True when mode is 'nf4' (backward compat)."""
        backend = self._make_backend()
        with patch.object(type(backend), '_get_quantization_mode', return_value='nf4'):
            assert backend._should_use_nvfp4() is True

    @patch.dict(os.environ, {"VRM_QUANTIZATION": ""})
    def test_should_use_nvfp4_returns_false_when_empty(self):
        """_should_use_nvfp4() returns False when no quantization."""
        backend = self._make_backend()
        with patch.object(type(backend), '_get_quantization_mode', return_value=''):
            assert backend._should_use_nvfp4() is False
