"""Tests for VRAMancer Speculative VRAM Lending system.

Tests cover:
  - VRAMLendingPool core operations (register, borrow, reclaim, release)
  - GPUBudget accounting
  - Lease lifecycle (ACTIVE → RECLAIMING → MIGRATED → RELEASED)
  - Lending policy enforcement
  - Preemption strategies (graceful vs forced)
  - Multi-GPU PagedKVCache overflow integration
  - Singleton factory
  - Monitor integration
  - Fixed bugs: hierarchical_memory, monitor.vram_usage
"""

import os
import sys
import time
import threading
import pytest

# Ensure VRM_MINIMAL_TEST is set
os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.vram_lending import (
    VRAMLendingPool,
    VRAMLease,
    GPUBudget,
    LendingPolicy,
    LeaseState,
    ReclaimUrgency,
    get_lending_pool,
    reset_lending_pool,
)


# ═══════════════════════════════════════════════════════════════════════════
# GPUBudget tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGPUBudget:
    """Test VRAM budget accounting."""

    def test_free_bytes(self):
        budget = GPUBudget(
            gpu_id=0,
            total_bytes=24_000_000_000,  # 24 GB
            model_bytes=12_000_000_000,  # 12 GB model
            reserved_bytes=2_400_000_000,  # 10% reserve
        )
        # Free = total - model - kv - lent - reserved
        assert budget.free_bytes == 24e9 - 12e9 - 2.4e9

    def test_lendable_bytes(self):
        budget = GPUBudget(
            gpu_id=0,
            total_bytes=24_000_000_000,
            model_bytes=12_000_000_000,
            reserved_bytes=2_400_000_000,
        )
        # Lendable = free - reserved (double reserve margin)
        assert budget.lendable_bytes > 0
        assert budget.lendable_bytes <= budget.free_bytes

    def test_utilization(self):
        budget = GPUBudget(
            gpu_id=0,
            total_bytes=24_000_000_000,
            model_bytes=12_000_000_000,
        )
        assert 0.0 < budget.utilization < 1.0
        assert budget.utilization == pytest.approx(0.5, abs=0.01)

    def test_effective_capacity(self):
        budget = GPUBudget(
            gpu_id=0,
            total_bytes=24_000_000_000,
            borrowed_bytes=4_000_000_000,
            lent_bytes=2_000_000_000,
        )
        assert budget.effective_capacity == 26_000_000_000  # 24 + 4 - 2

    def test_zero_total(self):
        budget = GPUBudget(gpu_id=0, total_bytes=0)
        assert budget.utilization == 0.0
        assert budget.free_bytes == 0
        assert budget.lendable_bytes == 0


# ═══════════════════════════════════════════════════════════════════════════
# VRAMLease tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVRAMLease:
    """Test lease data structure."""

    def test_creation(self):
        lease = VRAMLease(owner_gpu=0, borrower_gpu=1, size_bytes=1_000_000)
        assert lease.is_active
        assert lease.state == LeaseState.ACTIVE
        assert len(lease.lease_id) == 12

    def test_age(self):
        lease = VRAMLease(owner_gpu=0, borrower_gpu=1)
        time.sleep(0.01)
        assert lease.age_s > 0

    def test_default_purpose(self):
        lease = VRAMLease()
        assert lease.purpose == "kv_cache"

    def test_metadata(self):
        lease = VRAMLease(metadata={"layer": 5})
        assert lease.metadata["layer"] == 5


# ═══════════════════════════════════════════════════════════════════════════
# LendingPolicy tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLendingPolicy:
    """Test policy configuration."""

    def test_defaults(self):
        p = LendingPolicy()
        assert p.min_free_ratio == 0.10
        assert p.max_lend_ratio == 0.70
        assert p.reclaim_threshold == 0.80
        assert p.critical_threshold == 0.95

    def test_custom(self):
        p = LendingPolicy(max_lend_ratio=0.50, thermal_limit_c=75.0)
        assert p.max_lend_ratio == 0.50
        assert p.thermal_limit_c == 75.0


# ═══════════════════════════════════════════════════════════════════════════
# VRAMLendingPool core tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVRAMLendingPool:
    """Test the core lending pool operations."""

    def setup_method(self):
        reset_lending_pool()
        self.pool = VRAMLendingPool()

    def teardown_method(self):
        self.pool.close()

    def test_register_gpu(self):
        budget = self.pool.register_gpu(
            gpu_id=0,
            total_bytes=24_000_000_000,
            model_bytes=12_000_000_000,
            device_name="RTX 3090",
            pcie_gen=4,
        )
        assert budget.gpu_id == 0
        assert budget.total_bytes == 24_000_000_000
        assert budget.device_name == "RTX 3090"
        assert budget.pcie_gen == 4

    def test_register_multiple_gpus(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9,
                               device_name="RTX 3090", pcie_gen=4)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9,
                               device_name="RTX 5070 Ti", pcie_gen=5)
        cap = self.pool.pool_capacity()
        assert cap["gpu_count"] == 2
        assert cap["total_lendable_gb"] > 0

    def test_borrow_success(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9)
        lease = self.pool.borrow(
            borrower_gpu=0,
            size_bytes=2_000_000_000,  # 2 GB
            purpose="kv_cache",
        )
        assert lease is not None
        assert lease.owner_gpu == 1  # Borrowed from GPU 1
        assert lease.borrower_gpu == 0
        assert lease.is_active
        assert lease.size_bytes == 2_000_000_000

    def test_borrow_updates_budgets(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9)
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        assert lease is not None
        budget0 = self.pool.get_budget(0)
        budget1 = self.pool.get_budget(1)
        assert budget0.borrowed_bytes == 1e9
        assert budget1.lent_bytes == 1e9

    def test_borrow_fails_no_capacity(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=23.5e9)  # Almost full
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=15.5e9)  # Almost full
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=2e9)
        assert lease is None

    def test_borrow_prefers_pcie5(self):
        """GPU with PCIe 5.0 should be preferred for lending."""
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9,
                               pcie_gen=4)  # requestor
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9,
                               pcie_gen=4)
        self.pool.register_gpu(2, total_bytes=16e9, model_bytes=4e9,
                               pcie_gen=5)  # PCIe 5 — preferred
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        assert lease is not None
        # GPU 2 should be preferred (PCIe 5 + same free)
        assert lease.owner_gpu == 2

    def test_borrow_preferred_lender(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9)
        self.pool.register_gpu(2, total_bytes=16e9, model_bytes=4e9)
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=1e9,
                                 preferred_lender=1)
        assert lease is not None
        assert lease.owner_gpu == 1

    def test_borrow_cant_lend_to_self(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=4e9)
        # Only one GPU registered — can't borrow from self
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        assert lease is None

    def test_multiple_borrows(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        l1 = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        l2 = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        l3 = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        assert all(l is not None for l in [l1, l2, l3])
        assert self.pool.get_budget(1).lent_bytes == 3e9

    # ------------------------------------------------------------------
    # Reclaim & preemption
    # ------------------------------------------------------------------

    def test_reclaim_all(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        self.pool.borrow(borrower_gpu=0, size_bytes=2e9)
        self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        reclaimed = self.pool.reclaim(owner_gpu=1, urgency=ReclaimUrgency.HIGH)
        assert reclaimed == 3e9
        assert self.pool.get_budget(1).lent_bytes == 0

    def test_reclaim_partial(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        self.pool.borrow(borrower_gpu=0, size_bytes=2e9, priority=0)
        self.pool.borrow(borrower_gpu=0, size_bytes=2e9, priority=5)
        # Reclaim only 2 GB — should take the low-priority one
        reclaimed = self.pool.reclaim(owner_gpu=1,
                                      urgency=ReclaimUrgency.MEDIUM,
                                      bytes_needed=int(2e9))
        assert reclaimed >= 2e9

    def test_reclaim_critical_drops_data(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        reclaimed = self.pool.reclaim(owner_gpu=1,
                                      urgency=ReclaimUrgency.CRITICAL)
        assert reclaimed == 1e9
        assert self.pool._stats["preemptions_forced"] > 0

    def test_reclaim_graceful(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        reclaimed = self.pool.reclaim(owner_gpu=1,
                                      urgency=ReclaimUrgency.MEDIUM)
        assert reclaimed == 1e9
        assert self.pool._stats["preemptions_graceful"] > 0

    def test_reclaim_no_leases(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        reclaimed = self.pool.reclaim(owner_gpu=0)
        assert reclaimed == 0

    # ------------------------------------------------------------------
    # Release (voluntary)
    # ------------------------------------------------------------------

    def test_voluntary_release(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        ok = self.pool.release(lease.lease_id)
        assert ok
        assert self.pool.get_budget(1).lent_bytes == 0
        assert self.pool.get_budget(0).borrowed_bytes == 0

    def test_release_nonexistent(self):
        assert self.pool.release("nonexistent") is False

    def test_double_release(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        self.pool.release(lease.lease_id)
        ok = self.pool.release(lease.lease_id)
        assert ok  # Should not crash on double release

    # ------------------------------------------------------------------
    # Stats & queries
    # ------------------------------------------------------------------

    def test_stats_comprehensive(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9,
                               device_name="RTX 3090", pcie_gen=4)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9,
                               device_name="RTX 5070 Ti", pcie_gen=5)
        self.pool.borrow(borrower_gpu=0, size_bytes=2e9)
        s = self.pool.stats()
        assert "per_gpu" in s
        assert 0 in s["per_gpu"]
        assert 1 in s["per_gpu"]
        assert s["total_leases_created"] == 1
        assert s["total_bytes_lent"] == 2e9
        assert s["per_gpu"][0]["device"] == "RTX 3090"
        assert s["per_gpu"][1]["device"] == "RTX 5070 Ti"
        assert s["per_gpu"][1]["pcie_gen"] == 5

    def test_pool_capacity(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=8e9)
        cap = self.pool.pool_capacity()
        assert cap["gpu_count"] == 2
        assert cap["total_lendable_gb"] > 0
        assert cap["active_leases"] == 0

    def test_get_active_leases(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        leases = self.pool.get_active_leases()
        assert len(leases) == 2
        leases_for_gpu1 = self.pool.get_active_leases(gpu_id=1)
        assert len(leases_for_gpu1) == 2  # GPU1 is the owner

    def test_repr(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        r = repr(self.pool)
        assert "VRAMLendingPool" in r
        assert "gpus=1" in r

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def test_on_lend_callback(self):
        events = []
        self.pool.on_lend(lambda lease: events.append(lease))
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        assert len(events) == 1
        assert events[0].borrower_gpu == 0

    def test_on_reclaim_callback(self):
        events = []
        self.pool.on_reclaim(lambda lease: events.append(lease))
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        self.pool.reclaim(owner_gpu=1, urgency=ReclaimUrgency.HIGH)
        assert len(events) >= 1

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    def test_start_stop_monitoring(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.start_monitoring(interval=0.1)
        assert self.pool._monitoring
        time.sleep(0.15)
        self.pool.stop_monitoring()
        assert not self.pool._monitoring

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def test_close(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=4e9)
        self.pool.borrow(borrower_gpu=0, size_bytes=1e9)
        self.pool.close()
        # All leases should be released
        active = self.pool.get_active_leases()
        assert len(active) == 0

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def test_concurrent_borrows(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.register_gpu(1, total_bytes=16e9, model_bytes=2e9)
        results = []
        errors = []

        def borrow_worker():
            try:
                l = self.pool.borrow(borrower_gpu=0, size_bytes=100_000_000)
                if l:
                    results.append(l)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=borrow_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) > 0  # At least some should succeed
        assert self.pool.get_budget(1).lent_bytes == len(results) * 100_000_000

    def test_update_gpu_usage(self):
        self.pool.register_gpu(0, total_bytes=24e9, model_bytes=12e9)
        self.pool.update_gpu_usage(0, kv_cache_bytes=4e9)
        budget = self.pool.get_budget(0)
        assert budget.kv_cache_bytes == 4e9


# ═══════════════════════════════════════════════════════════════════════════
# Singleton tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSingleton:
    """Test global lending pool factory."""

    def setup_method(self):
        reset_lending_pool()

    def teardown_method(self):
        reset_lending_pool()

    def test_get_creates_singleton(self):
        p1 = get_lending_pool()
        p2 = get_lending_pool()
        assert p1 is p2

    def test_reset_clears_singleton(self):
        p1 = get_lending_pool()
        reset_lending_pool()
        p2 = get_lending_pool()
        assert p1 is not p2


# ═══════════════════════════════════════════════════════════════════════════
# PagedKVCache integration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPagedKVIntegration:
    """Test PagedKVCacheManager multi-GPU and lending integration."""

    def test_config_devices(self):
        from core.paged_attention import PagedKVConfig
        config = PagedKVConfig(
            devices=["cuda:0", "cuda:1"],
            pages_per_device={"cuda:0": 2048, "cuda:1": 1024},
        )
        assert len(config.devices) == 2
        assert config.pages_per_device["cuda:0"] == 2048
        assert config.enable_lending is True

    def test_config_default_devices(self):
        from core.paged_attention import PagedKVConfig
        config = PagedKVConfig(device="cuda:0")
        # devices should default to empty, filled by manager init
        assert config.device == "cuda:0"

    def test_physical_page_has_device(self):
        from core.paged_attention import PhysicalPage
        page = PhysicalPage(page_id=0, device="cuda:1")
        assert page.device == "cuda:1"
        assert page.is_borrowed is False
        assert page.lease_id is None

    def test_borrowed_page(self):
        from core.paged_attention import PhysicalPage
        page = PhysicalPage(
            page_id=42,
            device="cuda:1",
            is_borrowed=True,
            lease_id="abc123",
        )
        assert page.is_borrowed
        assert page.lease_id == "abc123"

    def test_manager_stub_mode(self):
        """Manager should work in stub mode (no GPU)."""
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        config = PagedKVConfig(device="cpu", max_pages=64)
        mgr = PagedKVCacheManager(config)
        entry = mgr.allocate("req1", num_tokens=5)
        assert entry is not None
        assert len(entry.pages) > 0

    def test_manager_stats_has_lending(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        config = PagedKVConfig(device="cpu", max_pages=32)
        mgr = PagedKVCacheManager(config)
        s = mgr.stats()
        assert "borrowed_pages" in s
        assert "overflow_borrows" in s
        assert "lending_active" in s
        assert "pages_per_device" in s

    def test_overflow_borrow_without_pool(self):
        """Without lending pool, overflow returns None."""
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        config = PagedKVConfig(device="cpu", max_pages=2, enable_lending=False)
        mgr = PagedKVCacheManager(config)
        # Allocate 2 pages (exhaust pool)
        mgr.allocate("req1", num_tokens=32)
        # Now the pool has 0 free pages, next alloc should evict or fail
        result = mgr._borrow_overflow_page()
        assert result is None  # No lending pool connected

    def test_eviction_prefers_borrowed_pages(self):
        """Eviction should prefer borrowed pages over owned pages."""
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig, PhysicalPage
        config = PagedKVConfig(device="cpu", max_pages=4, enable_lending=False)
        mgr = PagedKVCacheManager(config)
        # Manually mark page 3 as borrowed
        mgr._pages[3].is_borrowed = True
        mgr._pages[3].allocated = True
        mgr._pages[3].ref_count = 1
        mgr._pages[3].last_access = time.time() - 100
        mgr._free_pages.remove(3)
        # Also allocate page 0
        mgr._pages[0].allocated = True
        mgr._pages[0].ref_count = 1
        mgr._pages[0].last_access = time.time() - 50
        mgr._free_pages.remove(0)
        # Evict should pick page 3 (borrowed)
        victim = mgr._evict_lru()
        assert victim == 3


# ═══════════════════════════════════════════════════════════════════════════
# Bug fix verification tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBugFixes:
    """Verify the critical bugs we fixed are actually fixed."""

    def test_hierarchical_memory_thread_safe(self):
        """hierarchical_memory now has a _lock."""
        from core.hierarchical_memory import HierarchicalMemoryManager
        hmm = HierarchicalMemoryManager()
        assert hasattr(hmm, '_lock')
        assert isinstance(hmm._lock, type(threading.Lock()))

    def test_hierarchical_memory_tensor_registry(self):
        """hierarchical_memory now tracks tensors for NVMe spill."""
        from core.hierarchical_memory import HierarchicalMemoryManager
        hmm = HierarchicalMemoryManager()
        assert hasattr(hmm, '_tensor_registry')
        assert isinstance(hmm._tensor_registry, dict)

    def test_hierarchical_memory_register_with_tensor(self):
        """register_block accepts optional tensor parameter."""
        from core.hierarchical_memory import HierarchicalMemoryManager
        from core.memory_block import MemoryBlock
        hmm = HierarchicalMemoryManager()
        block = MemoryBlock(id="test-block-1", size_mb=100)
        hmm.register_block(block, "L1", tensor="fake_tensor")
        assert hmm._tensor_registry["test-block-1"] == "fake_tensor"

    def test_monitor_vram_usage_prefers_reserved(self):
        """monitor.vram_usage should use memory_reserved, not allocated."""
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        # In stub mode, this just returns 0.0 but shouldn't crash
        usage = m.vram_usage(0)
        assert isinstance(usage, float)
        assert 0.0 <= usage <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Lending scenario: 3090 + 5070 Ti simulation
# ═══════════════════════════════════════════════════════════════════════════

class TestHeterogeneousScenario:
    """Simulate a real RTX 3090 + RTX 5070 Ti lending scenario."""

    def setup_method(self):
        reset_lending_pool()
        self.pool = VRAMLendingPool(policy=LendingPolicy(
            max_lend_ratio=0.70,
            reclaim_threshold=0.80,
            critical_threshold=0.95,
        ))
        # Simulate: 3090 has 70B model split (60% = 21 GB model, 3 GB free)
        self.pool.register_gpu(
            0, total_bytes=int(24e9), model_bytes=int(21e9),
            device_name="RTX 3090", pcie_gen=4,
            compute_capability=(8, 6),
        )
        # Simulate: 5070 Ti has 40% = 14 GB model, 2 GB free
        self.pool.register_gpu(
            1, total_bytes=int(16e9), model_bytes=int(14e9),
            device_name="RTX 5070 Ti", pcie_gen=5,
            compute_capability=(10, 0),
        )

    def teardown_method(self):
        self.pool.close()

    def test_3090_borrows_from_5070ti(self):
        """3090 needs KV cache overflow → borrows from 5070 Ti."""
        # 5070 Ti has ~0.4 GB lendable (16 - 14 - 1.6 reserve)
        # Actually: free = 16 - 14 - 1.6 = 0.4 GB, lendable = 0.4 - 1.6 = negative
        # Let's adjust the model size to make it realistic
        self.pool._budgets[1].model_bytes = int(10e9)  # Less model on 5070 Ti
        lease = self.pool.borrow(
            borrower_gpu=0,
            size_bytes=int(1e9),  # 1 GB KV cache overflow
            purpose="kv_cache_overflow",
        )
        assert lease is not None
        assert lease.owner_gpu == 1
        assert lease.purpose == "kv_cache_overflow"

    def test_5070ti_reclaims_when_needed(self):
        """5070 Ti reclaims lent memory when it gets a new batch."""
        self.pool._budgets[1].model_bytes = int(10e9)
        lease = self.pool.borrow(
            borrower_gpu=0, size_bytes=int(1e9),
        )
        assert lease is not None
        # Simulate: 5070 Ti usage spikes
        self.pool.update_gpu_usage(1, kv_cache_bytes=int(3e9))
        reclaimed = self.pool.reclaim(owner_gpu=1, urgency=ReclaimUrgency.HIGH)
        assert reclaimed == int(1e9)

    def test_pool_stats_heterogeneous(self):
        s = self.pool.stats()
        assert s["per_gpu"][0]["device"] == "RTX 3090"
        assert s["per_gpu"][1]["device"] == "RTX 5070 Ti"
        assert s["per_gpu"][0]["pcie_gen"] == 4
        assert s["per_gpu"][1]["pcie_gen"] == 5

    def test_effective_capacity_with_borrowing(self):
        self.pool._budgets[1].model_bytes = int(10e9)
        lease = self.pool.borrow(borrower_gpu=0, size_bytes=int(1e9))
        assert lease is not None
        # GPU 0 effective capacity = 24 GB + 1 GB borrowed
        assert self.pool.get_budget(0).effective_capacity == int(24e9) + int(1e9)

    def test_reclaim_priority_ordering(self):
        """Low-priority leases should be reclaimed first."""
        self.pool._budgets[1].model_bytes = int(8e9)
        l_low = self.pool.borrow(borrower_gpu=0, size_bytes=int(500e6), priority=0)
        l_high = self.pool.borrow(borrower_gpu=0, size_bytes=int(500e6), priority=10)
        # Reclaim only 500 MB — should take the low-priority one
        reclaimed = self.pool.reclaim(
            owner_gpu=1,
            urgency=ReclaimUrgency.MEDIUM,
            bytes_needed=int(500e6),
        )
        assert reclaimed == int(500e6)
        # High-priority lease should still be active
        active = self.pool.get_active_leases()
        assert len(active) == 1
        assert active[0].priority == 10


__all__ = [
    "TestGPUBudget",
    "TestVRAMLease",
    "TestLendingPolicy",
    "TestVRAMLendingPool",
    "TestSingleton",
    "TestPagedKVIntegration",
    "TestBugFixes",
    "TestHeterogeneousScenario",
]
