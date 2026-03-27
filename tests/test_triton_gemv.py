"""Tests for core/triton_gemv_nvfp4.py — Triton GEMV LUT kernel for NVFP4."""
import os
import sys
import math
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    HAS_TORCH = hasattr(torch.nn, 'Module') and torch.nn.Module is not object
except (ImportError, AttributeError):
    HAS_TORCH = False

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()

# E2M1 FP4 lookup table (same as kernel)
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


class TestTritonGemvImport:
    """Module imports cleanly in any environment."""

    def test_module_imports(self):
        import core.triton_gemv_nvfp4 as mod
        assert hasattr(mod, "nvfp4_gemv")
        assert hasattr(mod, "_E2M1_VALUES")

    def test_lut_values_correct(self):
        from core.triton_gemv_nvfp4 import _E2M1_VALUES
        assert len(_E2M1_VALUES) == 16
        assert _E2M1_VALUES[0] == 0.0
        assert _E2M1_VALUES[6] == 4.0
        assert _E2M1_VALUES[15] == -6.0

    def test_has_triton_flag(self):
        from core.triton_gemv_nvfp4 import HAS_TRITON as flag
        assert isinstance(flag, bool)


@pytest.mark.skipif(not HAS_CUDA, reason="Requires CUDA GPU")
@pytest.mark.skipif(not HAS_TRITON, reason="Requires Triton")
class TestTritonGemvNumerics:
    """Numerical correctness tests (requires CUDA + Triton)."""

    @staticmethod
    def _make_fake_nvfp4(N, K, device="cuda"):
        """Create fake NVFP4 data with known values for verification."""
        # Random FP4 nibbles packed into uint8
        nib_even = torch.randint(0, 16, (N, K // 2), device=device, dtype=torch.uint8)
        nib_odd = torch.randint(0, 16, (N, K // 2), device=device, dtype=torch.uint8)
        w_qdata = (nib_odd << 4) | nib_even  # pack: low=even, high=odd

        # Random positive scales
        w_scale_row = torch.rand(N, K // 16, device=device, dtype=torch.float32) + 0.1

        # Build reference full-precision weight by dequantizing
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=device)
        weight_fp32 = torch.zeros(N, K, device=device, dtype=torch.float32)
        for byte_idx in range(K // 2):
            lo = nib_even[:, byte_idx]
            hi = nib_odd[:, byte_idx]
            weight_fp32[:, byte_idx * 2] = lut[lo.long()] * w_scale_row[:, byte_idx * 2 // 16]
            weight_fp32[:, byte_idx * 2 + 1] = lut[hi.long()] * w_scale_row[:, (byte_idx * 2 + 1) // 16]

        return w_qdata, w_scale_row, weight_fp32

    def test_basic_correctness(self):
        """GEMV output matches naive dequant + matmul."""
        from core.triton_gemv_nvfp4 import nvfp4_gemv

        N, K = 256, 128
        w_qdata, w_scale_row, weight_ref = self._make_fake_nvfp4(N, K)
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)

        # Triton kernel
        out_triton = nvfp4_gemv(x, w_qdata, w_scale_row)

        # Reference: x @ W_ref^T
        out_ref = (x.float() @ weight_ref.t()).to(x.dtype)

        torch.testing.assert_close(out_triton, out_ref, atol=1e-2, rtol=1e-2)

    def test_various_sizes(self):
        """Test with multiple N/K combinations."""
        from core.triton_gemv_nvfp4 import nvfp4_gemv

        for N, K in [(64, 64), (128, 256), (512, 128), (1024, 1024)]:
            w_qdata, w_scale_row, weight_ref = self._make_fake_nvfp4(N, K)
            x = torch.randn(1, K, device="cuda", dtype=torch.float16)

            out = nvfp4_gemv(x, w_qdata, w_scale_row)
            ref = (x.float() @ weight_ref.t()).to(x.dtype)

            torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2,
                                       msg=f"Failed for N={N}, K={K}")

    def test_dtype_preservation(self):
        """Output dtype matches input dtype."""
        from core.triton_gemv_nvfp4 import nvfp4_gemv

        N, K = 128, 128
        w_qdata, w_scale_row, _ = self._make_fake_nvfp4(N, K)

        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            x = torch.randn(1, K, device="cuda", dtype=dtype)
            out = nvfp4_gemv(x, w_qdata, w_scale_row)
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"

    def test_output_shape(self):
        """Output shape is [1, N]."""
        from core.triton_gemv_nvfp4 import nvfp4_gemv

        N, K = 512, 256
        w_qdata, w_scale_row, _ = self._make_fake_nvfp4(N, K)
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)

        out = nvfp4_gemv(x, w_qdata, w_scale_row)
        assert out.shape == (1, N)

    def test_1d_input(self):
        """Accepts [K] input and returns [N] output."""
        from core.triton_gemv_nvfp4 import nvfp4_gemv

        N, K = 128, 64
        w_qdata, w_scale_row, _ = self._make_fake_nvfp4(N, K)
        x = torch.randn(K, device="cuda", dtype=torch.float32)

        out = nvfp4_gemv(x, w_qdata, w_scale_row)
        assert out.shape == (N,)

    def test_no_nan(self):
        """Output contains no NaN values."""
        from core.triton_gemv_nvfp4 import nvfp4_gemv

        N, K = 1024, 512
        w_qdata, w_scale_row, _ = self._make_fake_nvfp4(N, K)
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)

        out = nvfp4_gemv(x, w_qdata, w_scale_row)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
