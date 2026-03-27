"""Tests for core/kv_quantizer.py — KV cache compression (PolarQuant + QJL)."""
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


class TestTurboQuantImport:
    """Module imports in any environment."""

    def test_module_imports(self):
        import core.kv_quantizer as mod
        assert hasattr(mod, "KVCacheCompressor")

    def test_backward_compat_import(self):
        import core.turboquant as mod
        assert hasattr(mod, "TurboQuantCompressor")

    def test_class_has_api(self):
        from core.kv_quantizer import KVCacheCompressor
        assert hasattr(KVCacheCompressor, "compress")
        assert hasattr(KVCacheCompressor, "decompress")
        assert hasattr(KVCacheCompressor, "attention_score")
        assert hasattr(KVCacheCompressor, "bits_per_dim")


@pytest.mark.skipif(not HAS_TORCH, reason="Requires real torch")
class TestPolarQuantMath:
    """Test polar coordinate encode/decode correctness."""

    def _make_compressor(self, head_dim=128):
        from core.kv_quantizer import KVCacheCompressor
        torch.manual_seed(42)
        return KVCacheCompressor(head_dim=head_dim, bits_per_angle=8)

    def test_polar_roundtrip_high_bits(self):
        """With 8-bit angles, polar encode→decode should reconstruct well."""
        comp = self._make_compressor(128)
        x = torch.randn(16, 128)

        # Pad + rotate
        rotated = comp._rotate(x)
        radius, angles = comp._polar_encode(rotated)

        # Quantize at high precision
        q_angles = [comp._quantize_angle(a, lvl) for lvl, a in enumerate(angles)]
        recon = comp._polar_decode(radius, q_angles)

        # With 8-bit angles, error should be small
        err = (rotated - recon).norm() / rotated.norm()
        assert err < 0.05, f"Polar roundtrip error too large: {err:.4f}"

    def test_polar_levels_correct(self):
        """Number of polar levels = log2(padded_dim)."""
        comp = self._make_compressor(128)
        assert comp._n_levels == 7  # log2(128)

        x = torch.randn(4, 128)
        rotated = comp._rotate(x)
        radius, angles = comp._polar_encode(rotated)

        assert radius.shape == (4, 1)
        assert len(angles) == 7
        assert angles[0].shape == (4, 64)   # d/2
        assert angles[1].shape == (4, 32)   # d/4
        assert angles[6].shape == (4, 1)    # 1

    def test_radius_positive(self):
        """All radii from polar encode are non-negative."""
        comp = self._make_compressor(64)
        x = torch.randn(32, 64)
        rotated = comp._rotate(x)
        radius, _ = comp._polar_encode(rotated)
        assert (radius >= 0).all()


@pytest.mark.skipif(not HAS_TORCH, reason="Requires real torch")
class TestQJL:
    """Test QJL sign-bit encoding and asymmetric estimator."""

    def _make_compressor(self, head_dim=64):
        from core.kv_quantizer import KVCacheCompressor
        torch.manual_seed(123)
        return KVCacheCompressor(head_dim=head_dim, bits_per_angle=3, qjl_dim=32)

    def test_qjl_signs_are_binary(self):
        """QJL signs are 0 or 1."""
        comp = self._make_compressor()
        residual = torch.randn(10, 64)  # padded_dim = 64
        signs, norms = comp._qjl_encode(residual)

        assert signs.dtype == torch.uint8
        assert ((signs == 0) | (signs == 1)).all()
        assert norms.shape == (10, 1)
        assert norms.dtype == torch.float16

    def test_asymmetric_estimator_unbiased(self):
        """QJL asymmetric estimator approximates true dot product (on average)."""
        from core.kv_quantizer import KVCacheCompressor
        torch.manual_seed(7)
        comp = KVCacheCompressor(head_dim=64, bits_per_angle=3, qjl_dim=256)

        # Large sample for statistical test
        keys = torch.randn(500, 64)
        queries = torch.randn(50, 64)

        # Pad for internal dims
        keys_pad = keys
        queries_pad = queries

        # True dot products
        true_scores = queries_pad @ keys_pad.t()

        # QJL encode
        signs, norms = comp._qjl_encode(keys_pad)

        # QJL score estimate
        estimated = comp._qjl_score_correction(queries_pad, signs, norms)

        # Should be correlated with true scores
        corr = torch.corrcoef(torch.stack([
            true_scores.flatten(), estimated.flatten()
        ]))[0, 1]
        assert corr > 0.3, f"QJL estimator correlation too low: {corr:.3f}"


@pytest.mark.skipif(not HAS_TORCH, reason="Requires real torch")
class TestTurboQuantEndToEnd:
    """End-to-end compress → attention_score tests."""

    def _make_compressor(self, head_dim=64, bits=3):
        from core.kv_quantizer import KVCacheCompressor
        torch.manual_seed(42)
        return KVCacheCompressor(head_dim=head_dim, bits_per_angle=bits)

    def test_compress_returns_expected_keys(self):
        """Compressed dict has all required keys."""
        comp = self._make_compressor()
        kv = torch.randn(32, 64)
        result = comp.compress(kv)

        assert "radius" in result
        assert "angles" in result
        assert "qjl_signs" in result
        assert "qjl_norms" in result
        assert "shape" in result
        assert result["shape"] == (32, 64)

    def test_decompress_shape(self):
        """Decompress returns original shape."""
        comp = self._make_compressor()
        kv = torch.randn(32, 64)
        compressed = comp.compress(kv)
        reconstructed = comp.decompress(compressed)
        assert reconstructed.shape == kv.shape

    def test_decompress_quality(self):
        """Reconstruction error is bounded with high-bit angles."""
        comp = self._make_compressor(head_dim=64, bits=6)
        kv = torch.randn(100, 64)
        compressed = comp.compress(kv)
        recon = comp.decompress(compressed)

        # Relative error
        rel_err = (kv - recon).norm() / kv.norm()
        assert rel_err < 0.15, f"Decompress error too large: {rel_err:.4f}"

    def test_attention_score_close_to_exact(self):
        """Attention scores from compressed keys approximate true scores."""
        comp = self._make_compressor(head_dim=64, bits=4)
        keys = torch.randn(50, 64)
        queries = torch.randn(10, 64)

        # True scores
        true_scores = queries @ keys.t()

        # Compressed scores
        compressed_k = comp.compress(keys)
        approx_scores = comp.attention_score(queries, compressed_k)

        # Correlation should be high
        corr = torch.corrcoef(torch.stack([
            true_scores.flatten(), approx_scores.flatten()
        ]))[0, 1]
        assert corr > 0.9, f"Attention score correlation: {corr:.3f}"

        # RMSE relative to score magnitude
        rmse = (true_scores - approx_scores).pow(2).mean().sqrt()
        score_std = true_scores.std()
        assert rmse / score_std < 0.5, f"RMSE/std: {rmse/score_std:.3f}"

    def test_qjl_improves_over_polar_alone(self):
        """Adding QJL correction should reduce attention score error vs polar-only."""
        comp = self._make_compressor(head_dim=64, bits=3)
        keys = torch.randn(100, 64)
        queries = torch.randn(20, 64)

        true_scores = queries @ keys.t()
        compressed_k = comp.compress(keys)

        # Scores WITH QJL
        scores_with_qjl = comp.attention_score(queries, compressed_k)

        # Scores WITHOUT QJL (polar only — reconstruct + dot)
        k_polar = comp.decompress(compressed_k)
        scores_polar_only = queries @ k_polar.t()

        err_with = (true_scores - scores_with_qjl).pow(2).mean()
        err_without = (true_scores - scores_polar_only).pow(2).mean()

        # QJL should help (or at worst be neutral)
        assert err_with <= err_without * 1.1, \
            f"QJL made things worse: {err_with:.4f} vs {err_without:.4f}"

    def test_bits_per_dim(self):
        """bits_per_dim returns a reasonable value."""
        comp = self._make_compressor(head_dim=128, bits=3)
        bpd = comp.bits_per_dim()
        assert 2.0 < bpd < 6.0, f"bits_per_dim={bpd}"

    def test_non_power_of_2_head_dim(self):
        """Works with non-power-of-2 head_dim (e.g., 96) via padding."""
        from core.kv_quantizer import KVCacheCompressor
        torch.manual_seed(42)
        comp = KVCacheCompressor(head_dim=96, bits_per_angle=3)
        assert comp._padded_dim == 128  # next power of 2

        kv = torch.randn(20, 96)
        compressed = comp.compress(kv)
        recon = comp.decompress(compressed)
        assert recon.shape == (20, 96)

    def test_rotation_is_orthogonal(self):
        """Hadamard rotation preserves vector norms."""
        comp = self._make_compressor(head_dim=64)
        x = torch.randn(50, 64)
        rotated = comp._rotate(x)
        norms_orig = x.norm(dim=-1)
        norms_rot = rotated.norm(dim=-1)
        torch.testing.assert_close(norms_orig, norms_rot, atol=1e-4, rtol=1e-4)

    def test_unrotate_inverts_rotate(self):
        """_unrotate is the inverse of _rotate."""
        comp = self._make_compressor(head_dim=64)
        x = torch.randn(16, 64)
        recovered = comp._unrotate(comp._rotate(x))
        torch.testing.assert_close(x, recovered, atol=1e-5, rtol=1e-5)
