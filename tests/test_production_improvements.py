"""Tests for production improvements: batching, GPU hot-plug, rebalancing, batch API."""
import os
import sys
import time
import json
import threading
import pytest

# Ensure test env
os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# Request Batching Tests
# ======================================================================

class TestInferenceBatcher:
    """Tests for core.api.batch_inference.InferenceBatcher."""

    def _dummy_generate(self, prompt, **kwargs):
        """Simulate generation with configurable delay."""
        time.sleep(0.01)
        max_tokens = kwargs.get("max_new_tokens", 10)
        return f"Response to: {prompt} (tokens={max_tokens})"

    def test_batcher_sync_fallback(self):
        """When batcher is not started, submit falls back to direct call."""
        from core.api.batch_inference import InferenceBatcher
        batcher = InferenceBatcher(generate_fn=self._dummy_generate)
        # Not started — should call generate_fn directly
        result = batcher.submit("Hello")
        assert "Response to: Hello" in result

    def test_batcher_start_stop(self):
        """Batcher lifecycle: start, submit, stop."""
        from core.api.batch_inference import InferenceBatcher
        batcher = InferenceBatcher(
            generate_fn=self._dummy_generate,
            max_batch=4,
            window_ms=20,
        )
        batcher.start()
        assert batcher._running is True

        result = batcher.submit("test prompt")
        assert "Response to: test prompt" in result

        batcher.stop()
        assert batcher._running is False

    def test_batcher_concurrent_requests(self):
        """Multiple concurrent requests are batched together."""
        from core.api.batch_inference import InferenceBatcher
        batcher = InferenceBatcher(
            generate_fn=self._dummy_generate,
            max_batch=8,
            window_ms=50,
        )
        batcher.start()

        results = [None] * 5
        errors = []

        def submit_request(idx):
            try:
                results[idx] = batcher.submit(f"Prompt {idx}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit_request, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        batcher.stop()

        assert not errors, f"Errors: {errors}"
        for i, r in enumerate(results):
            assert r is not None, f"Result {i} is None"
            assert f"Prompt {i}" in r

    def test_batcher_stats(self):
        """Stats are updated after processing."""
        from core.api.batch_inference import InferenceBatcher
        batcher = InferenceBatcher(
            generate_fn=self._dummy_generate,
            max_batch=4,
            window_ms=10,
        )
        batcher.start()

        batcher.submit("test1")
        batcher.submit("test2")
        time.sleep(0.2)  # let batch complete

        stats = batcher.stats
        assert stats["total_requests"] >= 2
        assert stats["total_batches"] >= 1
        assert stats["running"] is True

        batcher.stop()

    def test_batcher_error_propagation(self):
        """Errors from generate_fn are propagated to caller."""
        def failing_generate(prompt, **kwargs):
            raise ValueError("inference failed")

        from core.api.batch_inference import InferenceBatcher
        batcher = InferenceBatcher(
            generate_fn=failing_generate,
            window_ms=10,
        )
        batcher.start()

        with pytest.raises(ValueError, match="inference failed"):
            batcher.submit("test")

        batcher.stop()

    def test_batcher_kwargs_forwarded(self):
        """Generation kwargs are forwarded correctly."""
        received_kwargs = {}

        def capturing_generate(prompt, **kwargs):
            received_kwargs.update(kwargs)
            return "ok"

        from core.api.batch_inference import InferenceBatcher
        batcher = InferenceBatcher(
            generate_fn=capturing_generate,
            window_ms=10,
        )
        batcher.start()

        batcher.submit("test", max_new_tokens=42, temperature=0.7)
        time.sleep(0.2)

        assert received_kwargs.get("max_new_tokens") == 42
        assert received_kwargs.get("temperature") == 0.7

        batcher.stop()

    def test_batcher_pending_count(self):
        """pending_count reflects queue size."""
        from core.api.batch_inference import InferenceBatcher
        batcher = InferenceBatcher(
            generate_fn=self._dummy_generate,
            window_ms=10,
        )
        # Not started — pending stays 0
        assert batcher.pending_count == 0


# ======================================================================
# GPU Hot-Plug Monitor Tests
# ======================================================================

class TestGPUHotPlugMonitor:
    """Tests for core.monitor.GPUHotPlugMonitor."""

    def test_hotplug_init(self):
        """GPUHotPlugMonitor initializes with empty GPU list in test mode."""
        from core.monitor import GPUHotPlugMonitor
        hp = GPUHotPlugMonitor(interval=1.0)
        # In VRM_MINIMAL_TEST, no GPUs should be detected
        assert isinstance(hp.known_gpus, dict)

    def test_hotplug_start_stop(self):
        """Start/stop lifecycle works correctly."""
        from core.monitor import GPUHotPlugMonitor
        hp = GPUHotPlugMonitor(interval=0.5)
        hp.start()
        assert hp._running is True
        time.sleep(0.3)
        hp.stop()
        assert hp._running is False

    def test_hotplug_callbacks_registered(self):
        """Callbacks can be registered."""
        from core.monitor import GPUHotPlugMonitor
        hp = GPUHotPlugMonitor()
        events = []
        hp.on_add(lambda info: events.append(("add", info)))
        hp.on_remove(lambda info: events.append(("remove", info)))
        assert len(hp._on_add_callbacks) == 1
        assert len(hp._on_remove_callbacks) == 1

    def test_hotplug_linked_monitor(self):
        """Can be linked to a GPUMonitor instance."""
        from core.monitor import GPUHotPlugMonitor, GPUMonitor
        monitor = GPUMonitor()
        hp = GPUHotPlugMonitor(gpu_monitor=monitor)
        assert hp.gpu_monitor is monitor

    def test_hotplug_detect_gpus_minimal(self):
        """_detect_gpus returns empty list in minimal test mode."""
        from core.monitor import GPUHotPlugMonitor
        hp = GPUHotPlugMonitor()
        gpus = hp._detect_gpus()
        assert isinstance(gpus, list)
        # In VRM_MINIMAL_TEST, should be empty
        assert len(gpus) == 0

    def test_hotplug_known_gpus_threadsafe(self):
        """known_gpus property returns a copy (thread-safe)."""
        from core.monitor import GPUHotPlugMonitor
        hp = GPUHotPlugMonitor()
        gpus1 = hp.known_gpus
        gpus2 = hp.known_gpus
        assert gpus1 is not gpus2  # must be different objects (copies)

    def test_hotplug_double_start(self):
        """Starting twice is idempotent."""
        from core.monitor import GPUHotPlugMonitor
        hp = GPUHotPlugMonitor(interval=0.5)
        hp.start()
        hp.start()  # should be no-op
        assert hp._running is True
        hp.stop()


# ======================================================================
# Dynamic Rebalancing Tests
# ======================================================================

class TestDynamicRebalancing:
    """Tests for InferencePipeline dynamic rebalancing."""

    def test_rebalance_start_stop(self):
        """Rebalancing can be started and stopped."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(verbose=False, enable_metrics=False)
        # Without monitor/stream_manager, start should be a no-op
        pipe.start_rebalancing()
        assert pipe._rebalancing is False  # no subsystems

        pipe.stop_rebalancing()

    def test_rebalance_with_mock_subsystems(self):
        """Rebalancing works with mock monitor and stream_manager."""
        from core.inference_pipeline import InferencePipeline

        pipe = InferencePipeline(verbose=False, enable_metrics=False)

        # Mock monitor
        class MockMonitor:
            def detect_overload(self, threshold=None):
                return None  # no overload
        pipe.monitor = MockMonitor()

        # Mock stream manager
        class MockStreamManager:
            def swap_if_needed(self):
                return False
            def stop_monitoring(self):
                pass
        pipe.stream_manager = MockStreamManager()

        pipe.start_rebalancing(interval=0.2)
        assert pipe._rebalancing is True
        time.sleep(0.5)
        pipe.stop_rebalancing()
        assert pipe._rebalancing is False

    def test_rebalance_detects_overload(self):
        """Rebalancing triggers swap when overload is detected."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(verbose=False, enable_metrics=False)

        swapped = []

        class MockMonitor:
            def __init__(self):
                self._call_count = 0
            def detect_overload(self, threshold=None):
                self._call_count += 1
                return 0 if self._call_count <= 2 else None

        class MockStreamManager:
            def swap_if_needed(self):
                swapped.append(True)
                return True
            def stop_monitoring(self):
                pass

        pipe.monitor = MockMonitor()
        pipe.stream_manager = MockStreamManager()

        pipe.start_rebalancing(interval=0.1)
        time.sleep(0.5)
        pipe.stop_rebalancing()

        assert len(swapped) >= 1, "At least one swap should have been triggered"

    def test_pipeline_status_includes_subsystems(self):
        """Status dict includes all subsystem indicators."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(verbose=False, enable_metrics=False)
        pipe._loaded = True
        pipe.model_name = "test-model"
        status = pipe.status()
        assert "loaded" in status
        assert status["loaded"] is True
        assert status["model"] == "test-model"

    def test_pipeline_shutdown_idempotent(self):
        """Shutdown can be called multiple times safely."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(verbose=False, enable_metrics=False)
        pipe.shutdown()
        pipe.shutdown()  # no error


# ======================================================================
# Batch API Endpoint Tests
# ======================================================================

class TestBatchAPIEndpoint:
    """Tests for /v1/batch/completions endpoint."""

    @pytest.fixture
    def client(self):
        """Flask test client."""
        from core.production_api import create_app
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c

    def test_batch_missing_prompts(self, client):
        """Returns 400 when prompts field is missing."""
        resp = client.post('/v1/batch/completions',
                           json={},
                           headers={'Authorization': 'Bearer testtoken'})
        assert resp.status_code == 400
        data = resp.get_json()
        assert 'prompts' in data.get('error', '').lower()

    def test_batch_invalid_prompts_type(self, client):
        """Returns 400 when prompts is not a list."""
        resp = client.post('/v1/batch/completions',
                           json={'prompts': 'not a list'},
                           headers={'Authorization': 'Bearer testtoken'})
        assert resp.status_code == 400

    def test_batch_empty_prompts(self, client):
        """Returns 400 when prompts is an empty list."""
        resp = client.post('/v1/batch/completions',
                           json={'prompts': []},
                           headers={'Authorization': 'Bearer testtoken'})
        assert resp.status_code == 400

    def test_batch_endpoint_exists(self, client):
        """The /v1/batch/completions endpoint is registered."""
        # Without model loaded, we expect 400 (no model) not 404
        resp = client.post('/v1/batch/completions',
                           json={'prompts': ['hello']},
                           headers={'Authorization': 'Bearer testtoken'})
        assert resp.status_code != 404, "Endpoint should exist (not 404)"

    def test_api_status_includes_batch(self, client):
        """The /api/status endpoint lists the batch endpoint."""
        resp = client.get('/api/status',
                          headers={'Authorization': 'Bearer testtoken'})
        assert resp.status_code == 200
        data = resp.get_json()
        endpoints = data.get('endpoints', {})
        assert 'batch' in endpoints
        assert 'batch' in endpoints['batch'].lower() or '/v1/batch' in endpoints['batch']


# ======================================================================
# Auth Strong Tests (ensure_default_admin safety)
# ======================================================================

class TestAuthStrongSafety:
    """Validate auth_strong production safety."""

    def test_no_hardcoded_admin_password(self):
        """ensure_default_admin does NOT use 'admin' as password."""
        from core.auth_strong import ensure_default_admin, _USERS, verify_user
        _USERS.clear()  # reset
        ensure_default_admin()
        if "admin" in _USERS:
            # Password should NOT be 'admin'
            assert verify_user("admin", "admin") is False

    def test_production_refuses_default_admin(self):
        """In production mode, ensure_default_admin refuses to create user."""
        from core.auth_strong import ensure_default_admin, _USERS
        _USERS.clear()
        old = os.environ.get("VRM_PRODUCTION")
        os.environ["VRM_PRODUCTION"] = "1"
        try:
            ensure_default_admin()
            assert "admin" not in _USERS
        finally:
            if old is None:
                os.environ.pop("VRM_PRODUCTION", None)
            else:
                os.environ["VRM_PRODUCTION"] = old


# ======================================================================
# Import / Module Structure Tests
# ======================================================================

class TestModuleStructure:
    """Verify new modules are importable and well-structured."""

    def test_import_batch_inference(self):
        from core.api.batch_inference import InferenceBatcher
        assert InferenceBatcher is not None

    def test_import_gpu_hotplug(self):
        from core.monitor import GPUHotPlugMonitor
        assert GPUHotPlugMonitor is not None

    def test_batch_inference_all_exports(self):
        from core.api import batch_inference
        assert "InferenceBatcher" in batch_inference.__all__

    def test_monitor_exports_hotplug(self):
        """GPUHotPlugMonitor is accessible from core.monitor."""
        import core.monitor as mod
        assert hasattr(mod, "GPUHotPlugMonitor")
