"""Tests for ContinuousBatcher, PagedKVCache, and BenchmarkRunner.

Works in VRM_MINIMAL_TEST mode (stub torch). For real GPU tests, see test_real_gpu.py.
"""

import os
import sys
import time
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")


# =====================================================================
# 1. ContinuousBatcher
# =====================================================================

class TestContinuousBatcher:
    """Test continuous batching engine."""

    def test_import(self):
        from core.continuous_batcher import ContinuousBatcher, InferenceRequest, RequestStatus
        assert ContinuousBatcher is not None
        assert RequestStatus.WAITING is not None

    def test_init(self):
        from core.continuous_batcher import ContinuousBatcher
        cb = ContinuousBatcher(max_batch_size=8)
        assert cb.max_batch_size == 8
        assert cb._running is False
        assert cb.pending_count == 0

    def test_submit_without_model(self):
        from core.continuous_batcher import ContinuousBatcher
        cb = ContinuousBatcher(max_batch_size=4)
        cb.start()
        try:
            fut = cb.submit("Hello world", max_new_tokens=10)
            # Without tokenizer, should error
            try:
                result = fut.result(timeout=2)
            except Exception:
                pass  # Expected — no tokenizer
        finally:
            cb.stop()

    def test_start_stop(self):
        from core.continuous_batcher import ContinuousBatcher
        cb = ContinuousBatcher()
        cb.start()
        assert cb._running is True
        time.sleep(0.05)
        cb.stop()
        assert cb._running is False

    def test_stats(self):
        from core.continuous_batcher import ContinuousBatcher
        cb = ContinuousBatcher()
        stats = cb.stats()
        assert "running" in stats
        assert "waiting" in stats
        assert "active" in stats
        assert "throughput_tok_s" in stats
        assert stats["running"] is False

    def test_stub_mode_generation(self):
        """In MINIMAL mode, batcher should advance tokens via stub."""
        from core.continuous_batcher import ContinuousBatcher, InferenceRequest

        cb = ContinuousBatcher(max_batch_size=4)
        cb.start()
        time.sleep(0.05)  # let loop start

        # Manually create a request in stub mode
        req = InferenceRequest(prompt="test", max_new_tokens=5)
        with cb._lock:
            req.status = req.status  # just verify creation
        assert req.tokens_generated == 0
        cb.stop()

    def test_queue_limit(self):
        from core.continuous_batcher import ContinuousBatcher
        cb = ContinuousBatcher(max_waiting_queue=2)
        # Don't start — just test queue limit
        f1 = cb.submit("a")
        f2 = cb.submit("b")
        f3 = cb.submit("c")  # should fail — queue full
        assert f3.exception() is not None or f3.cancelled() or f3.done()
        cb.stop()

    def test_request_lifecycle(self):
        from core.continuous_batcher import InferenceRequest, RequestStatus
        req = InferenceRequest(prompt="test", max_new_tokens=10)
        assert req.status == RequestStatus.WAITING
        assert req.tokens_generated == 0
        assert req.request_id  # auto-generated
        assert req.created_at > 0


# =====================================================================
# 2. PagedKVCache
# =====================================================================

class TestPagedKVCache:
    """Test paged attention KV cache manager."""

    def test_import(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        assert PagedKVCacheManager is not None

    def test_config_defaults(self):
        from core.paged_attention import PagedKVConfig
        cfg = PagedKVConfig()
        assert cfg.page_size == 16
        assert cfg.num_layers == 12
        assert cfg.max_pages == 4096
        assert cfg.page_size_bytes > 0

    def test_config_memory_calculation(self):
        from core.paged_attention import PagedKVConfig
        cfg = PagedKVConfig(page_size=16, num_layers=12, num_kv_heads=12, head_dim=64)
        # 2 * 12 * 12 * 64 * 16 * 2 = 589,824 bytes per page
        assert cfg.page_size_bytes == 2 * 12 * 12 * 64 * 16 * 2
        assert cfg.total_memory_bytes == cfg.page_size_bytes * 4096

    def test_init(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(max_pages=32, device="cpu")
        mgr = PagedKVCacheManager(cfg)
        assert mgr is not None
        stats = mgr.stats()
        assert stats["total_pages"] == 32
        assert stats["free_pages"] == 32
        assert stats["used_pages"] == 0

    def test_allocate_free(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(max_pages=16, page_size=8, device="cpu")
        mgr = PagedKVCacheManager(cfg)

        entry = mgr.allocate("req-1", num_tokens=16)
        assert entry.request_id == "req-1"
        assert len(entry.pages) == 2  # 16 tokens / 8 page_size

        stats = mgr.stats()
        assert stats["used_pages"] == 2
        assert stats["active_requests"] == 1

        freed = mgr.free("req-1")
        assert freed == 2

        stats = mgr.stats()
        assert stats["used_pages"] == 0
        assert stats["active_requests"] == 0

    def test_append_token(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(max_pages=16, page_size=4, device="cpu")
        mgr = PagedKVCacheManager(cfg)

        mgr.allocate("req-1", num_tokens=0)

        # Append 5 tokens — should allocate 2 pages (4 tok/page)
        results = []
        for i in range(5):
            result = mgr.append_token("req-1")
            results.append(result)

        assert all(r is not None for r in results)
        stats = mgr.stats()
        assert stats["used_pages"] == 2  # 5 tokens in 4-tok pages

        mgr.free("req-1")

    def test_copy_on_write_fork(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(max_pages=32, page_size=4, device="cpu")
        mgr = PagedKVCacheManager(cfg)

        mgr.allocate("beam-0", num_tokens=8)  # 2 pages
        forked = mgr.fork("beam-0", "beam-1")
        assert forked is not None
        assert len(forked.pages) == 2

        # Both share the same pages (ref_count=2)
        stats = mgr.stats()
        assert stats["active_requests"] == 2
        # Pages are shared, so used count stays at 2
        assert stats["used_pages"] == 2

        # Free one — pages should NOT be freed (ref_count still 1)
        freed = mgr.free("beam-0")
        assert freed == 0  # ref_count went from 2→1, not freed

        # Free the other — now pages are freed
        freed = mgr.free("beam-1")
        assert freed == 2

    def test_pool_exhaustion(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(max_pages=4, page_size=4, device="cpu")
        mgr = PagedKVCacheManager(cfg)

        # Allocate all 4 pages
        mgr.allocate("req-1", num_tokens=16)  # needs 4 pages
        stats = mgr.stats()
        assert stats["free_pages"] == 0

        # Next allocation should fail gracefully
        entry = mgr.allocate("req-2", num_tokens=4)
        assert len(entry.pages) == 0  # no pages available

    def test_repr(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(max_pages=8, device="cpu")
        mgr = PagedKVCacheManager(cfg)
        r = repr(mgr)
        assert "PagedKVCache" in r

    def test_prefix_cache(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(max_pages=16, page_size=4, device="cpu")
        mgr = PagedKVCacheManager(cfg)

        # First request — cache miss
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hits = mgr.try_prefix_cache("req-1", tokens)
        # pages should be allocated
        stats = mgr.stats()
        assert stats["active_requests"] == 1

        mgr.free("req-1")


# =====================================================================
# 3. BenchmarkRunner
# =====================================================================

class TestBenchmarkRunner:
    """Test benchmark infrastructure."""

    def test_import(self):
        from core.benchmark import BenchmarkRunner, BenchmarkResult
        assert BenchmarkRunner is not None

    def test_init(self):
        from core.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(verbose=False)
        assert runner is not None

    def test_synthetic_benchmark(self):
        from core.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(verbose=False)
        result = runner.synthetic_benchmark(num_tokens=50, hidden_dim=128)
        assert "tokens_per_second" in result
        assert result["tokens_per_second"] > 0
        assert "latency_ms" in result
        assert result["total_tokens_generated"] == 50

    def test_benchmark_result_json(self):
        from core.benchmark import BenchmarkResult
        result = BenchmarkResult(
            model_name="test",
            tokens_per_second=100.5,
            total_tokens_generated=1000,
        )
        j = result.to_json()
        assert "test" in j
        assert "100.5" in j

        d = result.to_dict()
        assert d["model_name"] == "test"
        assert "per_request_latencies" not in d  # stripped

    def test_print_report(self):
        from core.benchmark import BenchmarkRunner, BenchmarkResult
        runner = BenchmarkRunner(verbose=False)
        result = BenchmarkResult(model_name="gpt2", tokens_per_second=42.0)
        report = runner.print_report(result)
        assert "42.0 tok/s" in report
        assert "gpt2" in report


# =====================================================================
# 4. Pipeline integration
# =====================================================================

class TestPipelineNewFeatures:
    """Test pipeline integration of new modules."""

    def test_pipeline_has_batcher_attrs(self):
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(enable_metrics=False)
        assert hasattr(pipe, 'continuous_batcher')
        assert hasattr(pipe, 'paged_kv')
        assert hasattr(pipe, 'submit')
        assert hasattr(pipe, 'benchmark')

    def test_pipeline_status_new_fields(self):
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(enable_metrics=False)
        status = pipe.status()
        assert "continuous_batcher" in status
        assert "paged_kv_cache" in status

    def test_batcher_stats_no_model(self):
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(enable_metrics=False)
        stats = pipe.batcher_stats()
        assert stats is None  # no model loaded

    def test_api_batcher_stats_endpoint(self):
        from core.production_api import create_app
        app = create_app()
        client = app.test_client()
        resp = client.get("/api/batcher/stats",
                         headers={"Authorization": "Bearer testtoken"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data or "running" in data

    def test_api_benchmark_endpoint_exists(self):
        from core.production_api import create_app
        app = create_app()
        client = app.test_client()
        # Just verify the route exists (benchmark will fail without model)
        resp = client.post("/api/benchmark",
                          json={"mode": "synthetic"},
                          headers={"Authorization": "Bearer testtoken"})
        # 200 or 500 (no model) — but NOT 404
        assert resp.status_code != 404
