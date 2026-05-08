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


# =============================================================================
# V6.D tests — PagedKVCacheManager lease lifecycle
# =============================================================================

def test_paged_kv_free_releases_borrowed_lease():
    """PagedKVCacheManager.free() releases the lending pool lease for borrowed pages."""
    from core.paged_attention import PagedKVCacheManager, PhysicalPage, PageTableEntry

    # Create manager bypassing __init__ to avoid torch dependencies
    manager = PagedKVCacheManager.__new__(PagedKVCacheManager)
    manager._lock = __import__('threading').Lock()
    manager._page_tables = {}
    manager._lending_leases = {}
    manager._free_pages = []
    manager._total_frees = 0
    manager._overflow_borrows = 0

    # Mock lending pool
    mock_pool = Mock()
    manager._lending_pool = mock_pool

    # Create a borrowed page at index 0 (page_id=0)
    page = PhysicalPage(
        page_id=0,
        ref_count=1,
        allocated=True,
        is_borrowed=True,
        lease_id="L123",
    )
    manager._pages = [page]  # Page at index 0

    # Register page in page table
    request_id = "req1"
    manager._page_tables[request_id] = PageTableEntry(
        request_id=request_id,
        pages=[0],  # Reference page_id 0
        num_tokens=10,
    )

    # Call free()
    manager.free(request_id)

    # Verify lease was released
    mock_pool.release.assert_called_once_with("L123")

    # Verify page state cleared
    assert page.is_borrowed is False
    assert page.lease_id is None
    assert page.ref_count == 0
    assert not page.allocated
    assert 0 in manager._free_pages


def test_paged_kv_free_no_release_when_not_borrowed():
    """PagedKVCacheManager.free() doesn't call pool.release() for local pages."""
    from core.paged_attention import PagedKVCacheManager, PhysicalPage, PageTableEntry

    # Create manager bypassing __init__
    manager = PagedKVCacheManager.__new__(PagedKVCacheManager)
    manager._lock = __import__('threading').Lock()
    manager._pages = []
    manager._page_tables = {}
    manager._lending_leases = {}
    manager._free_pages = []
    manager._total_frees = 0

    # Mock lending pool
    mock_pool = Mock()
    manager._lending_pool = mock_pool

    # Create a LOCAL page (not borrowed) at index 0
    page = PhysicalPage(
        page_id=0,
        ref_count=1,
        allocated=True,
        is_borrowed=False,  # NOT borrowed
        lease_id=None,
    )
    manager._pages = [page]  # Page at index 0

    # Register page in page table
    request_id = "req2"
    manager._page_tables[request_id] = PageTableEntry(
        request_id=request_id,
        pages=[0],  # Reference page_id 0
        num_tokens=8,
    )

    # Call free()
    manager.free(request_id)

    # Verify pool.release was NOT called
    mock_pool.release.assert_not_called()

    # Verify page was freed normally
    assert page.ref_count == 0
    assert not page.allocated
    assert 0 in manager._free_pages

def test_borrow_overflow_page_phase2_allocates_buffer_when_env_set(monkeypatch):
    """_borrow_overflow_page allocates empty buffer on lender GPU when VRM_KV_LEND=1."""
    from core.paged_attention import PagedKVCacheManager, PagedKVConfig

    # Set env var to enable Phase 2
    monkeypatch.setenv("VRM_KV_LEND", "1")

    # Create manager with minimal config
    manager = PagedKVCacheManager.__new__(PagedKVCacheManager)
    manager.config = PagedKVConfig(
        num_layers=4,
        num_kv_heads=8,
        page_size=16,
        head_dim=64,
        device="cuda:0",
    )
    manager._lock = __import__('threading').Lock()
    manager._pages = []
    manager._lending_leases = {}
    manager._overflow_borrows = 0
    manager._total_allocations = 0

    # Mock lending pool with borrow() and allocate_on_lease()
    mock_pool = Mock()
    mock_lease = Mock()
    mock_lease.lease_id = "L999"
    mock_lease.owner_gpu = 1
    mock_pool.borrow.return_value = mock_lease

    sentinel_tensor = Mock()
    mock_pool.allocate_on_lease.return_value = sentinel_tensor

    manager._lending_pool = mock_pool

    # Call _borrow_overflow_page
    page_id = manager._borrow_overflow_page()

    # Verify lease was acquired
    assert page_id is not None
    mock_pool.borrow.assert_called_once()

    # Verify allocate_on_lease was called with correct shape
    mock_pool.allocate_on_lease.assert_called_once()
    call_args = mock_pool.allocate_on_lease.call_args
    assert call_args[0][0] == mock_lease  # First arg is lease
    assert call_args[1]['shape'] == (4, 2, 8, 16, 64)  # (num_layers, 2, num_kv_heads, page_size, head_dim)

    # Verify borrowed_tensor was set on the page
    page = manager._pages[page_id]
    assert page.borrowed_tensor == sentinel_tensor


# =============================================================================
# V6.D Phase 3: cross-GPU staging in to_hf_cache()
# =============================================================================

def _make_manager_with_pool(num_layers=2, num_kv_heads=4, page_size=8, head_dim=16):
    """Helper: build a PagedKVCacheManager bypassing __init__ for unit tests."""
    import threading
    from core.paged_attention import PagedKVCacheManager, PagedKVConfig

    manager = PagedKVCacheManager.__new__(PagedKVCacheManager)
    manager.config = PagedKVConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        head_dim=head_dim,
        device="cuda:0",
    )
    manager._lock = threading.Lock()
    manager._pages = []
    manager._page_tables = {}
    manager._lending_leases = {}
    manager._free_pages = []
    manager._compressed_pages = {}
    manager._total_frees = 0
    manager._total_allocations = 0
    manager._overflow_borrows = 0
    manager._phase3_stages = 0
    manager._phase3_failures = 0
    manager._lending_pool = None
    manager._gpu_pool = None
    return manager


def test_to_hf_cache_phase3_stages_borrowed_pages_when_env_set(monkeypatch):
    """V6.D Phase 3: borrowed pages are staged via transfer_from_lease() when
    VRM_KV_LEND_ATTENTION=1, and merged into the returned past_key_values."""
    pytest.importorskip("torch")
    import torch
    from core.paged_attention import PhysicalPage, PageTableEntry

    monkeypatch.setenv("VRM_KV_LEND_ATTENTION", "1")

    manager = _make_manager_with_pool(num_layers=2, num_kv_heads=4, page_size=8, head_dim=16)

    # Build a fake _gpu_pool on CPU (simulating query GPU). Shape:
    # [max_pages, num_layers, 2(K/V), num_kv_heads, page_size, head_dim]
    pool_shape = (3, 2, 2, 4, 8, 16)
    manager._gpu_pool = torch.zeros(pool_shape)

    # Page 0: local — fill with 1.0 marker
    manager._gpu_pool[0] = 1.0
    # Page 1: borrowed — placeholder zeros in pool (real data is on lender)
    # Page 2: local — fill with 3.0 marker
    manager._gpu_pool[2] = 3.0

    manager._pages = [
        PhysicalPage(page_id=0, ref_count=1, allocated=True, is_borrowed=False),
        PhysicalPage(
            page_id=1, ref_count=1, allocated=True,
            is_borrowed=True, lease_id="L_phase3",
        ),
        PhysicalPage(page_id=2, ref_count=1, allocated=True, is_borrowed=False),
    ]

    # Mock lease + lending pool
    mock_lease = Mock()
    mock_lease.lease_id = "L_phase3"
    mock_lease.owner_gpu = 1
    manager._lending_leases["L_phase3"] = mock_lease

    # Borrowed page is referenced (must be non-None for Phase 3 to engage)
    manager._pages[1].borrowed_tensor = Mock()

    # Pool's transfer_from_lease returns a staged tensor on query GPU
    # filled with 2.0 — so we can verify the merge slotted it in.
    staged_tensor = torch.full((2, 2, 4, 8, 16), 2.0)
    mock_pool = Mock()
    mock_pool.transfer_from_lease.return_value = staged_tensor
    manager._lending_pool = mock_pool

    # Page table: request uses pages [0, 1, 2] in that order, 24 tokens total
    request_id = "req_phase3"
    manager._page_tables[request_id] = PageTableEntry(
        request_id=request_id,
        pages=[0, 1, 2],
        num_tokens=24,
    )

    # Call to_hf_cache
    pkv = manager.to_hf_cache(request_id)

    # Verify transfer_from_lease was called with the lease and query GPU 0
    mock_pool.transfer_from_lease.assert_called_once_with(mock_lease, 0)

    # Verify counters incremented
    assert manager._phase3_stages == 1
    assert manager._phase3_failures == 0

    # Verify shape: tuple of 2 layers, each (k, v), each [1, heads, 24, head_dim]
    assert pkv is not None
    assert len(pkv) == 2  # 2 layers
    for layer_kv in pkv:
        k, v = layer_kv
        assert k.shape == (1, 4, 24, 16)
        assert v.shape == (1, 4, 24, 16)
        # Tokens 0-7 from page 0 should be 1.0, 8-15 from staged page 1 should be 2.0,
        # 16-23 from page 2 should be 3.0
        assert torch.all(k[0, :, 0:8, :] == 1.0)
        assert torch.all(k[0, :, 8:16, :] == 2.0)
        assert torch.all(k[0, :, 16:24, :] == 3.0)


def test_to_hf_cache_phase3_disabled_by_default():
    """Without VRM_KV_LEND_ATTENTION=1, borrowed pages are NOT staged
    (original behavior preserved — borrowed slots return zeros)."""
    pytest.importorskip("torch")
    import torch
    from core.paged_attention import PhysicalPage, PageTableEntry

    # Ensure env var is OFF
    os.environ.pop("VRM_KV_LEND_ATTENTION", None)

    manager = _make_manager_with_pool(num_layers=1, num_kv_heads=2, page_size=4, head_dim=8)
    manager._gpu_pool = torch.zeros((2, 1, 2, 2, 4, 8))

    manager._pages = [
        PhysicalPage(
            page_id=0, ref_count=1, allocated=True,
            is_borrowed=True, lease_id="LX",
        ),
        PhysicalPage(page_id=1, ref_count=1, allocated=True, is_borrowed=False),
    ]
    manager._pages[0].borrowed_tensor = Mock()
    mock_pool = Mock()
    manager._lending_pool = mock_pool
    manager._lending_leases["LX"] = Mock(lease_id="LX", owner_gpu=1)

    manager._page_tables["r"] = PageTableEntry(
        request_id="r", pages=[0, 1], num_tokens=8,
    )

    pkv = manager.to_hf_cache("r")

    # transfer_from_lease MUST NOT have been called
    mock_pool.transfer_from_lease.assert_not_called()
    assert manager._phase3_stages == 0
    assert pkv is not None  # original code path still produces output


def test_to_hf_cache_phase3_handles_stage_failure(monkeypatch):
    """When transfer_from_lease() raises, Phase 3 falls back to the
    direct _gpu_pool read (degraded but non-fatal) and increments failure counter."""
    pytest.importorskip("torch")
    import torch
    from core.paged_attention import PhysicalPage, PageTableEntry

    monkeypatch.setenv("VRM_KV_LEND_ATTENTION", "1")

    manager = _make_manager_with_pool(num_layers=1, num_kv_heads=2, page_size=4, head_dim=8)
    manager._gpu_pool = torch.zeros((1, 1, 2, 2, 4, 8))

    manager._pages = [
        PhysicalPage(
            page_id=0, ref_count=1, allocated=True,
            is_borrowed=True, lease_id="LF",
        ),
    ]
    manager._pages[0].borrowed_tensor = Mock()

    mock_pool = Mock()
    mock_pool.transfer_from_lease.side_effect = RuntimeError("simulated transfer failure")
    manager._lending_pool = mock_pool
    manager._lending_leases["LF"] = Mock(lease_id="LF", owner_gpu=1)

    manager._page_tables["rf"] = PageTableEntry(
        request_id="rf", pages=[0], num_tokens=4,
    )

    pkv = manager.to_hf_cache("rf")

    # Failure counted, attempt was made
    assert manager._phase3_failures == 1
    assert manager._phase3_stages == 0
    mock_pool.transfer_from_lease.assert_called_once()
    # Output still produced (degraded path)
    assert pkv is not None


# =============================================================================
# V6.D Phase 4: write path routing (write_kv + from_hf_cache)
# =============================================================================

def test_write_kv_phase4_routes_borrowed_pages_to_lender(monkeypatch):
    """V6.D Phase 4: write_kv routes writes to page.borrowed_tensor when
    the page is borrowed AND VRM_KV_LEND_ATTENTION=1."""
    pytest.importorskip("torch")
    import torch
    from core.paged_attention import PhysicalPage

    monkeypatch.setenv("VRM_KV_LEND_ATTENTION", "1")

    manager = _make_manager_with_pool(num_layers=2, num_kv_heads=4, page_size=8, head_dim=16)
    # _gpu_pool stays zeros — the borrower-local backing
    manager._gpu_pool = torch.zeros((3, 2, 2, 4, 8, 16))
    manager._kv_compressor = None
    manager._pending_compress = []

    # borrowed_tensor on lender — also zeros, will receive the write
    borrowed = torch.zeros((2, 2, 4, 8, 16))
    manager._pages = [
        PhysicalPage(
            page_id=0, ref_count=1, allocated=True,
            is_borrowed=True, lease_id="LW",
            borrowed_tensor=borrowed,
        ),
    ]

    # Write a sentinel key/value to (page=0, layer=1, slot=3)
    key = torch.full((4, 16), 42.0)
    val = torch.full((4, 16), -7.0)
    manager.write_kv("req", layer_idx=1, page_id=0, slot=3, key=key, value=val)

    # Borrowed tensor must now contain the sentinel at the right slot
    assert torch.all(borrowed[1, 0, :, 3, :] == 42.0)
    assert torch.all(borrowed[1, 1, :, 3, :] == -7.0)
    # _gpu_pool must remain untouched (write was routed away)
    assert torch.all(manager._gpu_pool == 0.0)


def test_write_kv_phase4_disabled_by_default():
    """Without VRM_KV_LEND_ATTENTION=1, borrowed pages still receive
    writes in _gpu_pool (legacy behavior, data ends up on borrower)."""
    pytest.importorskip("torch")
    import torch
    from core.paged_attention import PhysicalPage

    os.environ.pop("VRM_KV_LEND_ATTENTION", None)

    manager = _make_manager_with_pool(num_layers=1, num_kv_heads=2, page_size=4, head_dim=8)
    manager._gpu_pool = torch.zeros((1, 1, 2, 2, 4, 8))
    manager._kv_compressor = None
    manager._pending_compress = []

    borrowed = torch.zeros((1, 2, 2, 4, 8))
    manager._pages = [
        PhysicalPage(
            page_id=0, ref_count=1, allocated=True,
            is_borrowed=True, lease_id="LD",
            borrowed_tensor=borrowed,
        ),
    ]

    key = torch.full((2, 8), 9.0)
    val = torch.full((2, 8), 11.0)
    manager.write_kv("req", layer_idx=0, page_id=0, slot=2, key=key, value=val)

    # Default-off: writes go to _gpu_pool, NOT to borrowed_tensor
    assert torch.all(borrowed == 0.0)
    assert torch.all(manager._gpu_pool[0, 0, 0, :, 2, :] == 9.0)
    assert torch.all(manager._gpu_pool[0, 0, 1, :, 2, :] == 11.0)


def test_write_kv_phase4_local_page_unaffected(monkeypatch):
    """Local (not borrowed) pages always go to _gpu_pool, even with
    VRM_KV_LEND_ATTENTION=1."""
    pytest.importorskip("torch")
    import torch
    from core.paged_attention import PhysicalPage

    monkeypatch.setenv("VRM_KV_LEND_ATTENTION", "1")

    manager = _make_manager_with_pool(num_layers=1, num_kv_heads=2, page_size=4, head_dim=8)
    manager._gpu_pool = torch.zeros((1, 1, 2, 2, 4, 8))
    manager._kv_compressor = None
    manager._pending_compress = []

    manager._pages = [
        PhysicalPage(page_id=0, ref_count=1, allocated=True, is_borrowed=False),
    ]

    key = torch.full((2, 8), 5.0)
    val = torch.full((2, 8), 3.0)
    manager.write_kv("req", layer_idx=0, page_id=0, slot=1, key=key, value=val)

    assert torch.all(manager._gpu_pool[0, 0, 0, :, 1, :] == 5.0)
    assert torch.all(manager._gpu_pool[0, 0, 1, :, 1, :] == 3.0)


def test_phase4_write_phase3_read_round_trip(monkeypatch):
    """End-to-end: write to a borrowed page (Phase 4), then read back
    via to_hf_cache (Phase 3 staging). The bytes must round-trip
    correctly through borrowed_tensor."""
    pytest.importorskip("torch")
    import torch
    from core.paged_attention import PhysicalPage, PageTableEntry

    monkeypatch.setenv("VRM_KV_LEND_ATTENTION", "1")

    manager = _make_manager_with_pool(num_layers=1, num_kv_heads=2, page_size=4, head_dim=8)
    manager._gpu_pool = torch.zeros((1, 1, 2, 2, 4, 8))
    manager._kv_compressor = None
    manager._pending_compress = []

    borrowed = torch.zeros((1, 2, 2, 4, 8))
    page = PhysicalPage(
        page_id=0, ref_count=1, allocated=True,
        is_borrowed=True, lease_id="LRT",
        borrowed_tensor=borrowed,
    )
    manager._pages = [page]

    # Phase 4 writes 4 tokens
    for slot in range(4):
        key = torch.full((2, 8), float(100 + slot))
        val = torch.full((2, 8), float(200 + slot))
        manager.write_kv("rt", layer_idx=0, page_id=0, slot=slot, key=key, value=val)

    # Phase 3 read: lending pool stages back the borrowed_tensor
    mock_lease = Mock(lease_id="LRT", owner_gpu=1)
    manager._lending_leases["LRT"] = mock_lease
    mock_pool = Mock()
    # transfer_from_lease returns the same borrowed tensor (simulating a
    # successful cross-GPU staging where data is preserved)
    mock_pool.transfer_from_lease.return_value = borrowed
    manager._lending_pool = mock_pool

    manager._page_tables["rt"] = PageTableEntry(
        request_id="rt", pages=[0], num_tokens=4,
    )

    pkv = manager.to_hf_cache("rt")

    assert pkv is not None
    k_out, v_out = pkv[0]
    # Shape: [1, num_kv_heads, num_tokens, head_dim] = [1, 2, 4, 8]
    assert k_out.shape == (1, 2, 4, 8)
    # Each slot's value: keys 100..103, vals 200..203
    for slot in range(4):
        assert torch.all(k_out[0, :, slot, :] == 100 + slot)
        assert torch.all(v_out[0, :, slot, :] == 200 + slot)

    # Phase 3 counter incremented exactly once (one borrowed page staged)
    assert manager._phase3_stages == 1

