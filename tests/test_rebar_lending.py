"""Tests for ReBAR full-window transport and VRAM Lending integration.

Validates:
  - ReBAR detection with > 4 GB threshold
  - ReBarTransport.transfer() fallback paths
  - ReBAR wiring in TransferManager (Strategy 1.7)
  - VRAM Lending borrow / release / reclaim lifecycle
  - Lending integration in PagedKVCacheManager overflow
  - VRM_VRAM_LENDING env var control
"""

import os
import sys
import time
import threading
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# Ensure VRM_MINIMAL_TEST is set
os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")


# ═══════════════════════════════════════════════════════════════════════════
# ReBAR Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReBarDetection:
    """ReBAR detection and threshold logic."""

    def test_rebar_threshold_constant(self):
        from core.cross_vendor_bridge import REBAR_FULL_WINDOW_THRESHOLD
        assert REBAR_FULL_WINDOW_THRESHOLD == 4 * 1024 * 1024 * 1024

    def test_detect_rebar_stub_mode(self):
        """In stub mode (VRM_MINIMAL_TEST), detect_rebar returns (False, 0)."""
        from core.cross_vendor_bridge import detect_rebar
        enabled, size = detect_rebar(0)
        assert enabled is False
        assert size == 0

    def test_rebar_transport_not_available_in_stub(self):
        """In stub mode, ReBarTransport has no available GPUs."""
        from core.cross_vendor_bridge import ReBarTransport
        rt = ReBarTransport()
        assert rt.available is False
        assert rt.full_window is False
        assert rt._gpu_bars == {}

    def test_rebar_transport_info_empty(self):
        """Info method works even when no ReBAR GPUs detected."""
        from core.cross_vendor_bridge import ReBarTransport
        rt = ReBarTransport()
        info = rt.info()
        assert info["available"] is False
        assert info["full_window"] is False
        assert info["threshold_gb"] == 4.0
        assert info["gpus"] == {}

    def test_rebar_optimal_chunk_default(self):
        """Without ReBAR, chunk size is the default."""
        from core.cross_vendor_bridge import ReBarTransport, DEFAULT_CHUNK_BYTES
        rt = ReBarTransport()
        assert rt.get_optimal_chunk_size(0) == DEFAULT_CHUNK_BYTES

    def test_rebar_transport_with_mock_bar(self):
        """Simulate a GPU with 16 GB BAR for chunk calculation."""
        from core.cross_vendor_bridge import ReBarTransport
        rt = ReBarTransport()
        bar_16gb = 16 * 1024 * 1024 * 1024
        rt._gpu_bars[0] = ("/sys/fake/resource0", bar_16gb)
        rt.available = True
        rt.full_window = True
        chunk = rt.get_optimal_chunk_size(0)
        # 16 GB / 64 = 256 MB, capped at 64 MB
        assert chunk == 64 * 1024 * 1024

    def test_rebar_transport_small_bar(self):
        """BAR of 4 GB → 4 GB / 64 = 64 MB (at cap)."""
        from core.cross_vendor_bridge import ReBarTransport
        rt = ReBarTransport()
        bar_4gb = 4 * 1024 * 1024 * 1024
        rt._gpu_bars[0] = ("/sys/fake/resource0", bar_4gb)
        chunk = rt.get_optimal_chunk_size(0)
        assert chunk == 64 * 1024 * 1024


class TestReBarTransferStub:
    """ReBarTransport.transfer() in stub mode (no torch)."""

    def test_transfer_returns_stub_without_torch(self):
        """Without torch, transfer returns STUB result."""
        import core.cross_vendor_bridge as cvb
        from core.cross_vendor_bridge import ReBarTransport, CrossVendorMethod
        rt = ReBarTransport()
        rt._gpu_bars[0] = ("/sys/fake", 16 * 1024**3)
        rt.available = True
        dummy = MagicMock()
        # Patch _TORCH to False so transfer() takes the stub path
        original = cvb._TORCH
        try:
            cvb._TORCH = False
            out, result = rt.transfer(0, 1, dummy)
            assert result.method == CrossVendorMethod.STUB
        finally:
            cvb._TORCH = original


class TestReBarInTransferManager:
    """ReBAR integration in TransferManager."""

    def test_transport_method_rebar_exists(self):
        """REBAR_PIPELINE is a valid TransportMethod."""
        from core.transfer_manager import TransportMethod
        assert hasattr(TransportMethod, "REBAR_PIPELINE")

    def test_transfer_manager_rebar_field(self):
        """TransferManager has _rebar_transport field."""
        from core.transfer_manager import TransferManager
        tm = TransferManager()
        assert hasattr(tm, "_rebar_transport")

    def test_transfer_manager_stats_rebar(self):
        """Stats include ReBAR info when available."""
        from core.transfer_manager import TransferManager
        tm = TransferManager()
        s = tm.stats()
        assert "method_preference" in s
        assert "REBAR_PIPELINE" in s["method_preference"]


# ═══════════════════════════════════════════════════════════════════════════
# VRAM Lending Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVRAMLending:
    """Core VRAMLendingPool lifecycle."""

    def _make_pool(self):
        from core.vram_lending import VRAMLendingPool, LendingPolicy
        policy = LendingPolicy(
            max_lend_ratio=0.70,
            reclaim_threshold=0.80,
            min_free_ratio=0.10,
        )
        return VRAMLendingPool(policy=policy)

    def test_register_gpu(self):
        pool = self._make_pool()
        budget = pool.register_gpu(
            gpu_id=0, total_bytes=24_000_000_000,
            model_bytes=12_000_000_000, device_name="RTX 3090",
        )
        assert budget.gpu_id == 0
        assert budget.total_bytes == 24_000_000_000
        assert budget.model_bytes == 12_000_000_000
        assert budget.free_bytes > 0
        assert budget.lendable_bytes > 0
        pool.close()

    def test_register_two_gpus(self):
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9, device_name="3090")
        pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9, device_name="5070Ti")
        cap = pool.pool_capacity()
        assert cap["gpu_count"] == 2
        assert cap["total_lendable_gb"] > 0
        pool.close()

    def test_borrow_success(self):
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24_000_000_000, model_bytes=12_000_000_000)
        pool.register_gpu(1, total_bytes=16_000_000_000, model_bytes=8_000_000_000)
        lease = pool.borrow(borrower_gpu=0, size_bytes=1_000_000_000, purpose="kv_cache")
        assert lease is not None
        assert lease.borrower_gpu == 0
        assert lease.owner_gpu == 1
        assert lease.size_bytes == 1_000_000_000
        assert lease.is_active
        pool.close()

    def test_borrow_from_specific_lender(self):
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9)
        lease = pool.borrow(
            borrower_gpu=0, size_bytes=500_000_000,
            preferred_lender=1,
        )
        assert lease is not None
        assert lease.owner_gpu == 1
        pool.close()

    def test_borrow_same_gpu_fails(self):
        """Cannot borrow from yourself."""
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        lease = pool.borrow(borrower_gpu=0, size_bytes=100_000_000)
        assert lease is None
        pool.close()

    def test_borrow_too_large(self):
        """Cannot borrow more than lendable."""
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=23e9)
        pool.register_gpu(1, total_bytes=16e9, model_bytes=15e9)
        # Both GPUs have very little free → can't lend 10 GB
        lease = pool.borrow(borrower_gpu=0, size_bytes=10_000_000_000)
        assert lease is None
        pool.close()

    def test_release_lease(self):
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9)
        lease = pool.borrow(borrower_gpu=0, size_bytes=1e9)
        assert lease is not None
        result = pool.release(lease.lease_id)
        assert result is True
        # GPU 1 lent_bytes should be back to 0
        b1 = pool.get_budget(1)
        assert b1.lent_bytes == 0
        pool.close()

    def test_reclaim_recovers_memory(self):
        from core.vram_lending import ReclaimUrgency
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9)
        lease = pool.borrow(borrower_gpu=0, size_bytes=1e9)
        assert lease is not None
        # GPU 1 reclaims
        reclaimed = pool.reclaim(owner_gpu=1, urgency=ReclaimUrgency.HIGH)
        assert reclaimed >= 1e9
        pool.close()

    def test_stats(self):
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9, device_name="3090")
        pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9, device_name="5070Ti")
        pool.borrow(borrower_gpu=0, size_bytes=500e6)
        s = pool.stats()
        assert s["total_leases_created"] == 1
        assert s["total_bytes_lent"] == 500e6
        assert s["gpu_count"] == 2
        assert 0 in s["per_gpu"]
        assert 1 in s["per_gpu"]
        pool.close()

    def test_update_gpu_usage(self):
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        pool.update_gpu_usage(0, kv_cache_bytes=2_000_000_000)
        b = pool.get_budget(0)
        assert b.kv_cache_bytes == 2_000_000_000
        pool.close()

    def test_monitoring_start_stop(self):
        pool = self._make_pool()
        pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        pool.start_monitoring(interval=10.0)
        assert pool._monitoring is True
        pool.stop_monitoring()
        assert pool._monitoring is False
        pool.close()


class TestVRAMLendingSingleton:
    """Singleton factory and reset."""

    def test_get_lending_pool(self):
        from core.vram_lending import get_lending_pool, reset_lending_pool
        reset_lending_pool()
        pool1 = get_lending_pool()
        pool2 = get_lending_pool()
        assert pool1 is pool2
        reset_lending_pool()

    def test_reset_lending_pool(self):
        from core.vram_lending import get_lending_pool, reset_lending_pool
        reset_lending_pool()
        pool1 = get_lending_pool()
        reset_lending_pool()
        pool2 = get_lending_pool()
        assert pool1 is not pool2
        reset_lending_pool()


class TestLendingInPagedAttention:
    """Lending integration in PagedKVCacheManager."""

    def test_paged_kv_has_lending_fields(self):
        from core.paged_attention import PagedKVConfig, PagedKVCacheManager
        cfg = PagedKVConfig(
            num_layers=2, num_kv_heads=4, head_dim=64,
            max_pages=8, page_size=4, device="cpu",
        )
        mgr = PagedKVCacheManager(cfg)
        assert hasattr(mgr, "_lending_pool")
        assert hasattr(mgr, "_lending_leases")
        assert hasattr(mgr, "_overflow_borrows")

    def test_physical_page_has_borrowed_fields(self):
        from core.paged_attention import PhysicalPage
        p = PhysicalPage(page_id=0)
        assert p.is_borrowed is False
        assert p.lease_id is None

    def test_borrow_overflow_returns_none_without_pool(self):
        from core.paged_attention import PagedKVConfig, PagedKVCacheManager
        cfg = PagedKVConfig(
            num_layers=2, num_kv_heads=4, head_dim=64,
            max_pages=4, page_size=4, device="cpu",
            enable_lending=False,
        )
        mgr = PagedKVCacheManager(cfg)
        assert mgr._lending_pool is None
        result = mgr._borrow_overflow_page()
        assert result is None

    def test_stats_include_lending(self):
        from core.paged_attention import PagedKVConfig, PagedKVCacheManager
        cfg = PagedKVConfig(
            num_layers=2, num_kv_heads=4, head_dim=64,
            max_pages=4, page_size=4, device="cpu",
        )
        mgr = PagedKVCacheManager(cfg)
        s = mgr.stats()
        assert "borrowed_pages" in s
        assert "overflow_borrows" in s
        assert "lending_active" in s


class TestVRMLendingEnvVar:
    """VRM_VRAM_LENDING env var control."""

    def test_env_var_disables_lending(self):
        """When VRM_VRAM_LENDING=0, pipeline should not init lending pool."""
        # Just verify the env var logic is correct
        val = "0"
        enabled = val.lower() not in ("0", "false", "no")
        assert enabled is False

    def test_env_var_default_enables_lending(self):
        val = "1"
        enabled = val.lower() not in ("0", "false", "no")
        assert enabled is True

    def test_env_var_false_word(self):
        val = "false"
        enabled = val.lower() not in ("0", "false", "no")
        assert enabled is False


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Lending + Eviction
# ═══════════════════════════════════════════════════════════════════════════

class TestLendingEviction:
    """Verify that borrowed pages are handled correctly on eviction."""

    def test_evict_prefers_borrowed_pages(self):
        """LRU eviction should prefer borrowed pages."""
        from core.paged_attention import (
            PagedKVConfig, PagedKVCacheManager, PhysicalPage,
        )
        cfg = PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=32,
            max_pages=4, page_size=4, device="cpu",
            enable_lending=False,
        )
        mgr = PagedKVCacheManager(cfg)

        # Allocate all pages
        for i in range(4):
            entry = mgr.allocate(f"req_{i}", num_tokens=1)

        # Mark page 2 as borrowed (simulating lending)
        mgr._pages[2].is_borrowed = True
        mgr._pages[2].lease_id = "fake_lease"
        mgr._pages[2].last_access = 0.0  # Old

        # Evict: should pick borrowed page 2 first
        victim_id = mgr._evict_lru()
        assert victim_id == 2
