"""V6.C — TransferManager-routed data plane for lending leases.

Tests the new transfer_into_lease() and transfer_from_lease() methods
that wire the lending pool to the TransferManager's strategy cascade
(Rust P2P, ReBAR, CPU-staged).
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch

from core.vram_lending import VRAMLendingPool, VRAMLease, LeaseState, LendingPolicy

# Check if we're in minimal test mode
_MINIMAL = os.environ.get('VRM_MINIMAL_TEST', '0') == '1'

# Check if we have real torch+CUDA for integration tests
_HAS_TORCH = False
_HAS_2GPU = False
if not _MINIMAL:
    try:
        import torch
        _HAS_TORCH = torch.cuda.is_available()
        _HAS_2GPU = _HAS_TORCH and torch.cuda.device_count() >= 2
    except ImportError:
        pass


# =============================================================================
# Unit tests (work without GPU, use mocks)
# =============================================================================

def test_transfer_into_lease_with_manager_mock():
    """transfer_into_lease calls TransferManager.send_tensor with correct args."""
    # Mock TransferManager
    mock_tm = Mock()
    mock_result_tensor = Mock()
    mock_tm.send_tensor.return_value = mock_result_tensor

    pool = VRAMLendingPool(transfer_manager=mock_tm)

    # Mock lease
    lease = VRAMLease(
        lease_id="test123",
        owner_gpu=1,
        borrower_gpu=0,
        size_bytes=1024,
        state=LeaseState.ACTIVE,
    )
    lease.tensor_ref = Mock()
    lease.tensor_ref.copy_ = Mock()

    # Mock source tensor
    src_tensor = Mock()
    src_tensor.device.index = 0

    # Call transfer_into_lease
    result = pool.transfer_into_lease(lease, src_tensor)

    # Verify TransferManager.send_tensor was called
    mock_tm.send_tensor.assert_called_once_with(0, 1, src_tensor)

    # Verify copy_ was called on lease.tensor_ref
    lease.tensor_ref.copy_.assert_called_once_with(mock_result_tensor)

    # Verify return value is the lease.tensor_ref
    assert result == lease.tensor_ref


def test_transfer_into_lease_fallback_no_manager():
    """Without a TransferManager, transfer_into_lease falls back to copy_()."""
    pool = VRAMLendingPool(transfer_manager=None)

    # Mock lease
    lease = VRAMLease(
        lease_id="test456",
        owner_gpu=1,
        borrower_gpu=0,
        size_bytes=1024,
        state=LeaseState.ACTIVE,
    )
    lease.tensor_ref = Mock()
    lease.tensor_ref.copy_ = Mock()

    # Mock source tensor
    src_tensor = Mock()
    src_tensor.device.index = 0

    # Call transfer_into_lease
    result = pool.transfer_into_lease(lease, src_tensor)

    # Verify direct copy_ was called (fallback path)
    lease.tensor_ref.copy_.assert_called_once_with(src_tensor)

    # Verify return value
    assert result == lease.tensor_ref


def test_transfer_into_lease_same_gpu_no_transfer():
    """transfer_into_lease on same GPU uses direct copy_ even with TransferManager."""
    mock_tm = Mock()

    pool = VRAMLendingPool(transfer_manager=mock_tm)

    # Mock lease on GPU 1
    lease = VRAMLease(
        lease_id="test789",
        owner_gpu=1,
        borrower_gpu=1,
        size_bytes=1024,
        state=LeaseState.ACTIVE,
    )
    lease.tensor_ref = Mock()
    lease.tensor_ref.copy_ = Mock()

    # Mock source tensor also on GPU 1
    src_tensor = Mock()
    src_tensor.device.index = 1

    # Call transfer_into_lease
    result = pool.transfer_into_lease(lease, src_tensor)

    # Verify TransferManager was NOT called (same GPU)
    mock_tm.send_tensor.assert_not_called()

    # Verify direct copy_ was called
    lease.tensor_ref.copy_.assert_called_once_with(src_tensor)

    assert result == lease.tensor_ref


def test_transfer_from_lease_with_manager_mock():
    """transfer_from_lease calls TransferManager.send_tensor to retrieve data."""
    mock_tm = Mock()
    mock_result_tensor = Mock()
    mock_tm.send_tensor.return_value = mock_result_tensor

    pool = VRAMLendingPool(transfer_manager=mock_tm)

    # Mock lease on GPU 1
    lease = VRAMLease(
        lease_id="test_read",
        owner_gpu=1,
        borrower_gpu=0,
        size_bytes=1024,
        state=LeaseState.ACTIVE,
    )
    lease.tensor_ref = Mock()

    # Call transfer_from_lease to materialize on GPU 0
    result = pool.transfer_from_lease(lease, dst_gpu=0)

    # Verify TransferManager.send_tensor was called
    mock_tm.send_tensor.assert_called_once_with(1, 0, lease.tensor_ref)

    # Verify return value
    assert result == mock_result_tensor


def test_transfer_from_lease_same_gpu_returns_ref():
    """transfer_from_lease on same GPU returns lease.tensor_ref directly."""
    pool = VRAMLendingPool(transfer_manager=Mock())

    lease = VRAMLease(
        lease_id="test_same",
        owner_gpu=1,
        borrower_gpu=1,
        size_bytes=1024,
        state=LeaseState.ACTIVE,
    )
    lease.tensor_ref = Mock()

    # Read back to same GPU
    result = pool.transfer_from_lease(lease, dst_gpu=1)

    # Should return tensor_ref directly without transfer
    assert result == lease.tensor_ref


def test_transfer_into_lease_inactive_lease_fails():
    """transfer_into_lease returns None for non-ACTIVE leases."""
    pool = VRAMLendingPool()

    lease = VRAMLease(
        lease_id="inactive",
        state=LeaseState.RELEASED,  # Not ACTIVE
    )
    lease.tensor_ref = Mock()

    src_tensor = Mock()
    src_tensor.device.index = 0

    result = pool.transfer_into_lease(lease, src_tensor)

    assert result is None


def test_transfer_into_lease_no_tensor_ref_fails():
    """transfer_into_lease returns None if lease.tensor_ref is None."""
    pool = VRAMLendingPool()

    lease = VRAMLease(
        lease_id="no_ref",
        state=LeaseState.ACTIVE,
    )
    lease.tensor_ref = None  # Not allocated yet

    src_tensor = Mock()
    src_tensor.device.index = 0

    result = pool.transfer_into_lease(lease, src_tensor)

    assert result is None


def test_transfer_from_lease_inactive_fails():
    """transfer_from_lease returns None for non-ACTIVE leases."""
    pool = VRAMLendingPool()

    lease = VRAMLease(
        lease_id="inactive_read",
        state=LeaseState.MIGRATED,  # Not ACTIVE
    )
    lease.tensor_ref = Mock()

    result = pool.transfer_from_lease(lease, dst_gpu=0)

    assert result is None


# =============================================================================
# Integration tests (require 2 CUDA GPUs)
# =============================================================================

@pytest.mark.skipif(not _HAS_2GPU, reason="needs 2 CUDA devices")
def test_transfer_into_lease_cross_gpu_real():
    """Writing to a lease on GPU1 from a tensor on GPU0 routes through
    TransferManager and the data lands correctly on GPU1."""
    import torch
    from core.transfer_manager import TransferManager

    tm = TransferManager(protocol="nccl", secure=False, verbose=False)
    pool = VRAMLendingPool(
        policy=LendingPolicy(buffer_prealloc_ratio=0.0),
        transfer_manager=tm,
    )

    pool.register_gpu(
        0, total_bytes=int(1e10), model_bytes=0,
        device_name="src", pcie_gen=4,
    )
    pool.register_gpu(
        1, total_bytes=int(1e10), model_bytes=0,
        device_name="dst", pcie_gen=4,
    )

    # Borrow from GPU 1 for GPU 0
    lease = pool.borrow(
        borrower_gpu=0,
        size_bytes=4 * 1024 * 1024,
        purpose="test",
        priority=1,
        preferred_lender=1,
    )
    assert lease is not None
    assert lease.owner_gpu == 1

    # Allocate on lease (creates empty buffer on GPU 1)
    leased = pool.allocate_on_lease(lease, shape=(1024, 1024), dtype=torch.float32)
    assert leased is not None
    assert leased.device.index == 1

    # Create source data on GPU 0
    src = torch.arange(
        1024 * 1024, dtype=torch.float32, device="cuda:0"
    ).view(1024, 1024)

    # Transfer into lease
    result = pool.transfer_into_lease(lease, src)
    assert result is not None
    assert result.device.index == 1
    assert torch.allclose(result.cpu(), src.cpu())

    # Read back to GPU 0
    back = pool.transfer_from_lease(lease, dst_gpu=0)
    assert back is not None
    assert back.device.index == 0
    assert torch.allclose(back.cpu(), src.cpu())

    # Cleanup
    pool.return_lease(lease)


@pytest.mark.skipif(not _HAS_2GPU, reason="needs 2 CUDA devices")
def test_transfer_into_lease_no_transfer_manager_fallback_real():
    """Without a TransferManager, transfer_into_lease falls back to copy_().

    This test verifies the fallback still works for same-GPU transfers.
    """
    import torch

    pool = VRAMLendingPool(policy=LendingPolicy(buffer_prealloc_ratio=0.0))
    pool.register_gpu(0, total_bytes=int(1e10), model_bytes=0, device_name="g0")
    pool.register_gpu(1, total_bytes=int(1e10), model_bytes=0, device_name="g1")

    lease = pool.borrow(
        borrower_gpu=0,
        size_bytes=1024 * 1024,
        purpose="test",
        priority=1,
        preferred_lender=1,
    )
    assert lease is not None

    leased = pool.allocate_on_lease(lease, shape=(256, 256), dtype=torch.float32)
    assert leased is not None
    assert leased.device.index == 1

    # Same-GPU transfer (create tensor on GPU 1, copy to lease on GPU 1)
    src_same = torch.ones(256, 256, dtype=torch.float32, device="cuda:1")
    result = pool.transfer_into_lease(lease, src_same)
    assert result is not None
    assert torch.allclose(result, src_same)

    pool.return_lease(lease)
