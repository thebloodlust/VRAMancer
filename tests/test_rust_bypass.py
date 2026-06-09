"""Tests for Rust CUDA bypass module (vramancer_rust).

Tests both the core Rust functions and the integration with TransferManager.
"""
import os
import sys
import pytest

# Ensure project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Module-level availability check
# ---------------------------------------------------------------------------

try:
    import vramancer_rust as vr
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# When VRM_REQUIRE_RUST=1 (set by the GPU CI job), a missing/uncompiled Rust
# module is a HARD FAILURE rather than a silent skip — this prevents shipping
# or demoing without the CUDA bypass actually loaded into the interpreter.
_REQUIRE_RUST = os.environ.get("VRM_REQUIRE_RUST", "") in ("1", "true", "yes")
if _REQUIRE_RUST and not _RUST_AVAILABLE:
    raise RuntimeError(
        "VRM_REQUIRE_RUST=1 but `import vramancer_rust` failed. "
        "Build it with: cd rust_core && CUDA_PATH=/usr maturin develop "
        "--release --features cuda"
    )

try:
    import torch
    _TORCH_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() >= 2
except Exception:
    _TORCH_AVAILABLE = False

needs_rust = pytest.mark.skipif(not _RUST_AVAILABLE, reason="vramancer_rust not compiled")
needs_gpu = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="Need >=2 CUDA GPUs")


@pytest.mark.skipif(not _REQUIRE_RUST, reason="enforcement only when VRM_REQUIRE_RUST=1")
def test_rust_required_when_enforced():
    """Hard gate: when VRM_REQUIRE_RUST=1, Rust must import AND expose CUDA FFI."""
    assert _RUST_AVAILABLE, "vramancer_rust must be importable when VRM_REQUIRE_RUST=1"
    assert vr.cuda_available(), "vramancer_rust CUDA FFI must be available"


# ---------------------------------------------------------------------------
# Core Rust function tests (no GPU required)
# ---------------------------------------------------------------------------

class TestRustCoreFunctions:
    """Test Rust functions that don't need GPU."""

    @needs_rust
    def test_module_imports(self):
        assert hasattr(vr, 'detect_best_transport')
        assert hasattr(vr, 'sign_payload_fast')
        assert hasattr(vr, 'verify_hmac_fast')
        assert hasattr(vr, 'generate_xor_parity')
        assert hasattr(vr, 'repair_xor_shard')

    @needs_rust
    def test_detect_best_transport(self):
        tier = vr.detect_best_transport()
        assert tier is not None

    @needs_rust
    def test_hmac_sign_verify(self):
        data = b"test payload for hmac"
        key = b"secret-key-123"
        sig = vr.sign_payload_fast(key, data)
        assert isinstance(sig, bytes)
        assert len(sig) == 32

        assert vr.verify_hmac_fast(key, data, sig) is True
        assert vr.verify_hmac_fast(key, b"tampered", sig) is False

    @needs_rust
    def test_hmac_batch_verify(self):
        key = b"batch-key"
        items = [
            (b"payload1", vr.sign_payload_fast(key, b"payload1")),
            (b"payload2", vr.sign_payload_fast(key, b"payload2")),
            (b"payload3", b"\x00" * 32),  # bad signature
        ]
        results = vr.verify_hmac_batch(key, items)
        assert results == [True, True, False]

    @needs_rust
    def test_xor_parity(self):
        shard_a = b"\x01\x02\x03\x04"
        shard_b = b"\x05\x06\x07\x08"
        parity = vr.generate_xor_parity([shard_a, shard_b])
        assert isinstance(parity, bytes)
        assert len(parity) == 4

        # Reconstruct shard_b from shard_a + parity
        healed = vr.repair_xor_shard([shard_a], parity)
        assert healed == shard_b

    @needs_rust
    def test_xor_parity_empty(self):
        parity = vr.generate_xor_parity([])
        assert parity == b""

    @needs_rust
    def test_parity_deprecated_aliases(self):
        """Old names must still work (backward compat)."""
        shard_a = b"\x01\x02\x03\x04"
        shard_b = b"\x05\x06\x07\x08"
        parity = vr.generate_holographic_parity([shard_a, shard_b])
        assert vr.heal_holograph([shard_a], parity) == shard_b

    @needs_rust
    def test_staged_gpu_transfer_registered(self):
        """staged_gpu_transfer function exists in module."""
        assert hasattr(vr, 'staged_gpu_transfer')

    @needs_rust
    def test_direct_vram_copy_registered(self):
        """direct_vram_copy function exists in module."""
        assert hasattr(vr, 'direct_vram_copy')

    @needs_rust
    def test_async_gpu_transfer_registered(self):
        """async_gpu_transfer function exists in module."""
        assert hasattr(vr, 'async_gpu_transfer')

    @needs_rust
    def test_gpu_pipeline_registered(self):
        """GpuPipeline class exists in module."""
        assert hasattr(vr, 'GpuPipeline')
        assert hasattr(vr, 'direct_vram_copy')

    @needs_rust
    def test_inject_to_vram_ptr_registered(self):
        """inject_to_vram_ptr function exists in module."""
        assert hasattr(vr, 'inject_to_vram_ptr')

    @needs_rust
    def test_function_count(self):
        """Module should have at least 17 public functions/classes."""
        public = [x for x in dir(vr) if not x.startswith('_')]
        assert len(public) >= 17


# ---------------------------------------------------------------------------
# GPU-dependent tests (require >=2 CUDA GPUs)
# ---------------------------------------------------------------------------

class TestRustGPUTransfer:
    """Test Rust CUDA driver API functions on real GPUs."""

    @needs_rust
    @needs_gpu
    def test_direct_vram_copy_small(self):
        """cuMemcpyDtoD works for small tensors."""
        import torch
        a = torch.randn(1024, device='cuda:0')
        b = torch.empty_like(a, device='cuda:1')
        torch.cuda.synchronize()

        nbytes = a.element_size() * a.nelement()
        result = vr.direct_vram_copy(a.data_ptr(), b.data_ptr(), nbytes)
        torch.cuda.synchronize()

        assert result is True
        assert torch.allclose(a.cpu(), b.cpu())

    @needs_rust
    @needs_gpu
    def test_direct_vram_copy_large(self):
        """cuMemcpyDtoD works for larger tensors (10 MB)."""
        import torch
        n = 10 * 1024 * 1024 // 4  # 10 MB float32
        a = torch.randn(n, device='cuda:0')
        b = torch.empty_like(a, device='cuda:1')
        torch.cuda.synchronize()

        nbytes = a.element_size() * a.nelement()
        vr.direct_vram_copy(a.data_ptr(), b.data_ptr(), nbytes)
        torch.cuda.synchronize()

        assert torch.allclose(a.cpu(), b.cpu())

    @needs_rust
    @needs_gpu
    def test_staged_gpu_transfer(self):
        """Double-buffered staged transfer with data integrity."""
        import torch
        n = 4 * 1024 * 1024 // 4  # 4 MB float32
        a = torch.randn(n, device='cuda:0')
        b = torch.empty_like(a, device='cuda:1')
        torch.cuda.synchronize()

        nbytes = a.element_size() * a.nelement()
        result = vr.staged_gpu_transfer(
            a.data_ptr(), b.data_ptr(), nbytes, 0, 1, 1 * 1024 * 1024  # 1MB chunks
        )

        assert result is True
        assert torch.allclose(a.cpu(), b.cpu())

    @needs_rust
    @needs_gpu
    def test_inject_to_vram_ptr(self):
        """cuMemcpyHtoD injecting bytes into pre-allocated VRAM tensor."""
        import torch
        import numpy as np

        host_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
        dst = torch.empty(4, dtype=torch.float32, device='cuda:0')
        torch.cuda.synchronize()

        vr.inject_to_vram_ptr(host_data, dst.data_ptr())
        torch.cuda.synchronize()

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(dst.cpu(), expected)

    @needs_rust
    @needs_gpu
    def test_reverse_direction(self):
        """Transfer works GPU 1 -> GPU 0 as well."""
        import torch
        a = torch.randn(2048, device='cuda:1')
        b = torch.empty_like(a, device='cuda:0')
        torch.cuda.synchronize()

        nbytes = a.element_size() * a.nelement()
        vr.direct_vram_copy(a.data_ptr(), b.data_ptr(), nbytes)
        torch.cuda.synchronize()

        assert torch.allclose(a.cpu(), b.cpu())

    @needs_rust
    @needs_gpu
    def test_gpu_pipeline_transfer(self):
        """GpuPipeline persistent async pipeline: correct data."""
        import torch
        # Ensure CUDA context initialized on both GPUs
        _ = torch.zeros(1, device='cuda:0')
        _ = torch.zeros(1, device='cuda:1')
        torch.cuda.synchronize()

        pipe = vr.GpuPipeline(0, 1, 4)  # 4 MB chunks
        n = 5 * 1024 * 1024 // 4  # 5 MB of float32
        src = torch.randn(n, device='cuda:0')
        dst = torch.empty(n, device='cuda:1')
        torch.cuda.synchronize()

        pipe.transfer(src.data_ptr(), dst.data_ptr(), n * 4)
        torch.cuda.synchronize()

        assert torch.allclose(src.cpu(), dst.cpu())

    @needs_rust
    @needs_gpu
    def test_async_gpu_transfer_correctness(self):
        """async_gpu_transfer non-persistent: correct data."""
        import torch
        n = 2 * 1024 * 1024 // 4  # 2 MB of float32
        src = torch.randn(n, device='cuda:0')
        dst = torch.empty(n, device='cuda:1')
        torch.cuda.synchronize()

        vr.async_gpu_transfer(
            src.data_ptr(), dst.data_ptr(), n * 4, 0, 1, 1 * 1024 * 1024
        )
        torch.cuda.synchronize()

        assert torch.allclose(src.cpu(), dst.cpu())


# ---------------------------------------------------------------------------
# TransferManager integration tests
# ---------------------------------------------------------------------------

class TestTransferManagerRustIntegration:
    """Test that TransferManager uses Rust bypass when available."""

    def test_transfer_manager_imports(self):
        from core.transfer_manager import TransferManager
        assert TransferManager is not None

    def test_transfer_manager_init(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager()
        assert tm is not None

    @needs_rust
    @needs_gpu
    def test_transfer_uses_rust_bypass(self):
        """send_activation should use Rust DtoD when available."""
        import torch
        from core.transfer_manager import TransferManager

        tm = TransferManager(verbose=True)
        tensor = torch.randn(1024, device='cuda:0')
        torch.cuda.synchronize()

        result = tm.send_activation(0, 1, tensor)
        assert result is not None
        assert result.duration_s >= 0
