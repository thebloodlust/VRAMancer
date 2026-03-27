"""Stress test for VRAMLendingPool — concurrent borrow/reclaim/release.

Tests the lending pool under multi-threaded pressure to verify:
  - No deadlocks under concurrent access
  - Budget accounting stays consistent
  - Reclaim works under pressure
  - No lease leaks after shutdown

Markers: @pytest.mark.integration (needs threading, no GPU required)
"""
import os
import sys
import time
import threading
import pytest

# Ensure test env
os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.vram_lending import VRAMLendingPool, LendingPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(num_gpus=4, total_per_gpu=24 * 1024**3):
    """Create a lending pool with ``num_gpus`` fake GPUs."""
    policy = LendingPolicy()
    policy.max_lend_ratio = 0.70
    policy.reclaim_threshold = 0.80
    pool = VRAMLendingPool(policy=policy)
    for i in range(num_gpus):
        pool.register_gpu(
            gpu_id=i,
            total_bytes=total_per_gpu,
            model_bytes=int(total_per_gpu * 0.30),  # 30% used by model
            device_name=f"FakeGPU-{i}",
            pcie_gen=4,
            compute_capability=(8, 6),
        )
    return pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLendingStress:
    """Multi-threaded stress tests for VRAMLendingPool."""

    def test_concurrent_borrow_release(self):
        """Hammer borrow/release from 8 threads, verify no lease leak."""
        pool = _make_pool(num_gpus=4)
        num_threads = 8
        ops_per_thread = 50
        errors = []
        lease_ids = []
        lock = threading.Lock()

        def _worker(thread_id):
            borrower = thread_id % 4
            for i in range(ops_per_thread):
                try:
                    lease = pool.borrow(
                        borrower_gpu=borrower,
                        size_bytes=1 * 1024**2,  # 1 MB
                        purpose="stress_test",
                        priority=thread_id % 5,
                    )
                    if lease is not None:
                        with lock:
                            lease_ids.append(lease.lease_id)
                        # Simulate some work
                        time.sleep(0.001)
                        pool.release(lease.lease_id)
                except Exception as e:
                    with lock:
                        errors.append((thread_id, i, str(e)))

        threads = [
            threading.Thread(target=_worker, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during stress test: {errors[:5]}"

        # All leases should be released — check budgets are clean
        stats = pool.stats()
        assert stats["active_leases"] == 0, (
            f"Lease leak: {stats['active_leases']} still active"
        )

    def test_concurrent_borrow_reclaim(self):
        """Borrow from many threads while reclaim fires on the lender."""
        pool = _make_pool(num_gpus=2)
        errors = []
        reclaim_results = []
        lease_refs = []
        lock = threading.Lock()

        def _borrower():
            for _ in range(30):
                try:
                    lease = pool.borrow(
                        borrower_gpu=1,
                        size_bytes=2 * 1024**2,
                        purpose="kv_cache",
                    )
                    if lease:
                        with lock:
                            lease_refs.append(lease.lease_id)
                        time.sleep(0.002)
                        pool.release(lease.lease_id)
                except Exception as e:
                    with lock:
                        errors.append(("borrow", str(e)))

        def _reclaimer():
            for _ in range(10):
                try:
                    freed = pool.reclaim(owner_gpu=0, bytes_needed=5 * 1024**2)
                    with lock:
                        reclaim_results.append(freed)
                    time.sleep(0.005)
                except Exception as e:
                    with lock:
                        errors.append(("reclaim", str(e)))

        workers = [threading.Thread(target=_borrower) for _ in range(4)]
        workers.append(threading.Thread(target=_reclaimer))

        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=30)

        assert not errors, f"Errors: {errors[:5]}"

    def test_budget_consistency(self):
        """After many borrow/release cycles, budgets must balance."""
        pool = _make_pool(num_gpus=3)
        total_per_gpu = 24 * 1024**3

        for _ in range(100):
            lease = pool.borrow(
                borrower_gpu=2,
                size_bytes=10 * 1024**2,  # 10 MB
                purpose="test",
            )
            if lease:
                pool.release(lease.lease_id)

        stats = pool.stats()
        assert stats["active_leases"] == 0

        # Each GPU's lent_bytes and borrowed_bytes should be 0
        for gpu_id in range(3):
            budget = pool._budgets.get(gpu_id)
            if budget:
                assert budget.lent_bytes == 0, (
                    f"GPU {gpu_id} lent_bytes={budget.lent_bytes} (should be 0)"
                )
                assert budget.borrowed_bytes == 0, (
                    f"GPU {gpu_id} borrowed_bytes={budget.borrowed_bytes} (should be 0)"
                )

    def test_exhaust_and_recover(self):
        """Exhaust lending capacity, then release all and borrow again."""
        pool = _make_pool(num_gpus=2, total_per_gpu=1 * 1024**3)  # 1 GB each
        leases = []

        # Borrow until exhausted
        for _ in range(200):
            lease = pool.borrow(
                borrower_gpu=1,
                size_bytes=5 * 1024**2,  # 5 MB
                purpose="exhaust",
            )
            if lease is None:
                break
            leases.append(lease)

        assert len(leases) > 0, "Should have borrowed at least some"

        # Release all
        for lease in leases:
            pool.release(lease.lease_id)

        # Should be able to borrow again
        lease = pool.borrow(
            borrower_gpu=1,
            size_bytes=5 * 1024**2,
            purpose="recovery",
        )
        assert lease is not None, "Should be able to borrow after releasing all"
        pool.release(lease.lease_id)

    def test_shutdown_cleans_leases(self):
        """Shutdown must clean up all active leases."""
        pool = _make_pool(num_gpus=2)

        # Create some leases without releasing them
        leases = []
        for _ in range(10):
            lease = pool.borrow(
                borrower_gpu=1,
                size_bytes=1 * 1024**2,
                purpose="leaked",
            )
            if lease:
                leases.append(lease)

        assert len(leases) > 0

        # Close should clean them
        pool.close()

        stats = pool.stats()
        assert stats["active_leases"] == 0, "close() should clean all leases"

    def test_high_contention_no_deadlock(self):
        """16 threads fighting for the same 2 GPUs — must not deadlock."""
        pool = _make_pool(num_gpus=2)
        barrier = threading.Barrier(16, timeout=10)
        completed = threading.Event()
        deadlocked = []

        def _fighter(tid):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                return
            for _ in range(20):
                borrower = tid % 2
                lease = pool.borrow(
                    borrower_gpu=borrower,
                    size_bytes=512 * 1024,  # 512 KB
                    purpose="contention",
                )
                if lease:
                    pool.release(lease.lease_id)

        threads = [threading.Thread(target=_fighter, args=(t,)) for t in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            if t.is_alive():
                deadlocked.append(t.name)

        assert not deadlocked, f"Deadlocked threads: {deadlocked}"
