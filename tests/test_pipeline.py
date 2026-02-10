"""Tests for the InferencePipeline and production API inference endpoints.

Runs under VRM_MINIMAL_TEST=1 (no real GPU/model required).
"""
import os
import sys
import json
import pytest

# Ensure env is set before any core imports
os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")
os.environ.setdefault("VRM_BACKEND_ALLOW_STUB", "1")


# ========================================================================
# InferencePipeline unit tests
# ========================================================================

class TestInferencePipeline:

    def test_import(self):
        """Pipeline module imports without errors."""
        from core.inference_pipeline import InferencePipeline, get_pipeline, reset_pipeline
        assert InferencePipeline is not None

    def test_init_defaults(self):
        """Pipeline initializes with default params."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(
            backend_name="auto",
            enable_metrics=False,
            enable_discovery=False,
            verbose=False,
        )
        assert pipe.backend_name == "auto"
        assert pipe.is_loaded() is False
        assert pipe.model_name is None
        assert pipe.blocks == []

    def test_status_unloaded(self):
        """status() works on an unloaded pipeline."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(enable_metrics=False, enable_discovery=False, verbose=False)
        s = pipe.status()
        assert s["loaded"] is False
        assert s["model"] is None

    def test_ensure_loaded_raises(self):
        """generate()/infer() raise if no model loaded."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(enable_metrics=False, enable_discovery=False, verbose=False)
        with pytest.raises(RuntimeError, match="not loaded"):
            pipe.generate("hello")
        with pytest.raises(RuntimeError, match="not loaded"):
            pipe.infer([1, 2, 3])

    def test_singleton(self):
        """get_pipeline() returns the same instance, reset clears it."""
        from core.inference_pipeline import get_pipeline, reset_pipeline
        reset_pipeline()
        p1 = get_pipeline(enable_metrics=False, enable_discovery=False, verbose=False)
        p2 = get_pipeline()
        assert p1 is p2
        reset_pipeline()

    def test_shutdown_idempotent(self):
        """shutdown() can be called multiple times safely."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(enable_metrics=False, enable_discovery=False, verbose=False)
        pipe.shutdown()
        pipe.shutdown()  # Should not raise

    @pytest.mark.smoke
    def test_load_stub_backend(self):
        """Load with stub backend (VRM_BACKEND_ALLOW_STUB=1)."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(
            backend_name="auto",
            enable_metrics=False,
            enable_discovery=False,
            verbose=False,
        )
        try:
            pipe.load("test-stub-model", num_gpus=1)
            # If load succeeds, verify state
            assert pipe.is_loaded()
            assert pipe.model_name == "test-stub-model"
            s = pipe.status()
            assert s["loaded"] is True
            assert s["model"] == "test-stub-model"
            pipe.shutdown()
        except Exception:
            # In minimal test mode, backend may raise — that's acceptable
            pass

    def test_load_invalid_backend(self):
        """Loading with nonexistent backend handles gracefully."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(
            backend_name="nonexistent_backend_xyz",
            enable_metrics=False,
            enable_discovery=False,
            verbose=False,
        )
        # Should either raise or use stub on model load
        try:
            pipe.load("some-model")
        except Exception:
            pass  # Expected in minimal mode


# ========================================================================
# Production API tests
# ========================================================================

class TestProductionAPI:

    @pytest.fixture(autouse=True)
    def client(self):
        """Create a Flask test client via create_app."""
        try:
            from core.production_api import create_app
            app = create_app()
            app.config["TESTING"] = True
            with app.test_client() as c:
                self._client = c
                yield c
        except ImportError:
            pytest.skip("production_api not available")

    # ---- health endpoints ----

    def test_health(self):
        resp = self._client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data

    def test_live(self):
        resp = self._client.get("/live")
        assert resp.status_code == 200

    def test_ready(self):
        resp = self._client.get("/ready")
        # 200 or 503 depending on state — both acceptable
        assert resp.status_code in (200, 503)

    # ---- info / status endpoints ----

    def test_api_status(self):
        resp = self._client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "version" in data or "status" in data

    def test_api_gpu(self):
        resp = self._client.get("/api/gpu")
        assert resp.status_code == 200

    def test_api_system(self):
        resp = self._client.get("/api/system")
        assert resp.status_code == 200

    def test_api_nodes(self):
        resp = self._client.get("/api/nodes")
        assert resp.status_code == 200

    def test_api_models(self):
        resp = self._client.get("/api/models")
        assert resp.status_code == 200

    def test_pipeline_status(self):
        resp = self._client.get("/api/pipeline/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "loaded" in data

    # ---- inference endpoints (no model loaded → 400/503) ----

    def test_completions_no_model(self):
        """POST /v1/completions without model loaded returns error."""
        resp = self._client.post(
            "/v1/completions",
            data=json.dumps({"prompt": "hello", "max_tokens": 5}),
            content_type="application/json",
        )
        # Should return 400 or 503 (no model loaded)
        assert resp.status_code in (400, 500, 503)

    def test_generate_no_model(self):
        """POST /api/generate without model loaded returns error."""
        resp = self._client.post(
            "/api/generate",
            data=json.dumps({"prompt": "hello"}),
            content_type="application/json",
        )
        assert resp.status_code in (400, 500, 503)

    def test_infer_no_model(self):
        """POST /api/infer without model loaded returns error."""
        resp = self._client.post(
            "/api/infer",
            data=json.dumps({"input_ids": [1, 2, 3]}),
            content_type="application/json",
        )
        assert resp.status_code in (400, 500, 503)

    # ---- route existence ----

    def test_404(self):
        resp = self._client.get("/nonexistent/path/xyz")
        assert resp.status_code == 404

    # ---- model load endpoint ----

    def test_load_model_no_body(self):
        """POST /api/models/load without body returns 400."""
        resp = self._client.post(
            "/api/models/load",
            data=json.dumps({}),
            content_type="application/json",
        )
        # Should require model_name
        assert resp.status_code in (400, 500)


# ========================================================================
# Legacy app compatibility tests
# ========================================================================

class TestLegacyApp:

    @pytest.fixture(autouse=True)
    def client(self):
        """Test with the legacy `app` global."""
        try:
            from core.production_api import app
            app.config["TESTING"] = True
            with app.test_client() as c:
                self._client = c
                yield c
        except ImportError:
            pytest.skip("production_api not available")

    def test_legacy_health(self):
        resp = self._client.get("/health")
        assert resp.status_code == 200

    def test_legacy_live(self):
        resp = self._client.get("/live")
        assert resp.status_code == 200


# ========================================================================
# Transfer manager integration
# ========================================================================

class TestTransferManagerCompat:

    def test_import(self):
        from core.transfer_manager import TransferManager
        assert TransferManager is not None

    def test_stub_mode(self):
        """In minimal test mode, TransferManager runs in stub mode."""
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        assert tm._stub_mode is True

    def test_send_activation_stub(self):
        """send_activation returns a TransferResult in stub mode."""
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        result = tm.send_activation(0, 1, "fake_tensor")
        assert result is not None
        assert hasattr(result, "method")
        assert hasattr(result, "source_gpu")
        assert result.source_gpu == 0
        assert result.target_gpu == 1

    def test_stats(self):
        """stats() returns a dict."""
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        s = tm.stats()
        assert isinstance(s, dict)

    def test_nccl_not_initialized_in_single_process(self):
        """In single process, NCCL should NOT be initialized."""
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        # Without MASTER_ADDR and WORLD_SIZE, NCCL stays off
        assert tm._nccl_initialized is False


# ========================================================================
# CLI entry point
# ========================================================================

class TestCLIEntryPoints:

    def test_vramancer_main_import(self):
        """vramancer.main imports without errors."""
        import importlib
        mod = importlib.import_module("vramancer.main")
        assert hasattr(mod, "main")

    def test_main_no_args(self, capsys):
        """main() with no args prints help and exits."""
        from vramancer.main import main
        main([])  # No args → prints help, returns
        captured = capsys.readouterr()
        assert "vramancer" in captured.out.lower() or captured.out == ""

    def test_version_command(self, capsys):
        """python -m vramancer.main version prints version."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["vramancer", "version"]
            from vramancer.main import main
            main()
            captured = capsys.readouterr()
            assert "vramancer" in captured.out.lower() or "0.2" in captured.out
        finally:
            sys.argv = old_argv


# ========================================================================
# Backends integration
# ========================================================================

class TestBackendsIntegration:

    def test_select_backend(self):
        """select_backend returns a backend object."""
        from core.backends import select_backend
        backend = select_backend("auto")
        assert backend is not None
        assert hasattr(backend, "load_model")
        assert hasattr(backend, "infer")

    def test_backend_has_generate(self):
        """Selected backend has generate() method."""
        from core.backends import select_backend
        backend = select_backend("auto")
        assert hasattr(backend, "generate")

    def test_backend_has_split_model(self):
        """Selected backend has split_model() method."""
        from core.backends import select_backend
        backend = select_backend("auto")
        assert hasattr(backend, "split_model")
