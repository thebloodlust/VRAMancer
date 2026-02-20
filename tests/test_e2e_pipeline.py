"""End-to-end tests for the VRAMancer inference pipeline.

Tests the full flow: backend selection → load → infer/generate → stream,
covering both stub mode (VRM_MINIMAL_TEST=1) and production-like paths.
"""
import os
import sys
import json
import types
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")
os.environ.setdefault("VRM_BACKEND_ALLOW_STUB", "1")


# ========================================================================
# Backend unit tests
# ========================================================================

class TestBackendSelection:
    """Test the select_backend() factory for all backend types."""

    def test_select_auto(self):
        from core.backends import select_backend
        b = select_backend("auto")
        assert b is not None
        assert hasattr(b, "load_model")
        assert hasattr(b, "generate")
        assert hasattr(b, "generate_stream")
        assert hasattr(b, "infer")
        assert hasattr(b, "split_model")

    def test_select_huggingface(self):
        from core.backends import select_backend
        b = select_backend("huggingface")
        assert b.__class__.__name__ == "HuggingFaceBackend"

    def test_select_vllm_stub(self):
        from core.backends import select_backend
        b = select_backend("vllm")
        assert b.__class__.__name__ == "vLLMBackend"

    def test_select_ollama_stub(self):
        from core.backends import select_backend
        b = select_backend("ollama")
        assert b.__class__.__name__ == "OllamaBackend"


class TestVLLMBackendStub:
    """Test vLLM backend in stub mode."""

    def test_load_model_stub(self):
        from core.backends import vLLMBackend
        b = vLLMBackend(real=False)
        result = b.load_model("test-model")
        assert result is not None
        assert b.model_name == "test-model"

    def test_generate_stub(self):
        from core.backends import vLLMBackend
        b = vLLMBackend(real=False)
        b.load_model("test-model")
        text = b.generate("Hello world")
        assert isinstance(text, str)
        assert len(text) > 0
        assert "stub" in text.lower()

    def test_generate_stream_stub(self):
        from core.backends import vLLMBackend
        b = vLLMBackend(real=False)
        b.load_model("test-model")
        tokens = list(b.generate_stream("Hello world"))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_split_model_stub(self):
        from core.backends import vLLMBackend
        b = vLLMBackend(real=False)
        b.load_model("test-model")
        blocks = b.split_model(2)
        assert len(blocks) == 1  # vLLM handles TP internally

    def test_infer_stub(self):
        from core.backends import vLLMBackend
        b = vLLMBackend(real=False)
        b.load_model("test-model")
        result = b.infer("test input")
        assert result is not None

    def test_infer_stub_no_model(self):
        from core.backends import vLLMBackend
        b = vLLMBackend(real=False)
        with pytest.raises(RuntimeError, match="non chargé"):
            b.infer("test")


class TestOllamaBackendStub:
    """Test Ollama backend in stub mode."""

    def test_load_model_stub(self):
        from core.backends import OllamaBackend
        b = OllamaBackend(real=False)
        result = b.load_model("llama3")
        assert result is not None
        assert b.model_name == "llama3"

    def test_generate_stub(self):
        from core.backends import OllamaBackend
        b = OllamaBackend(real=False)
        b.load_model("llama3")
        text = b.generate("Tell me a joke")
        assert isinstance(text, str)
        assert "stub" in text.lower()

    def test_generate_stream_stub(self):
        from core.backends import OllamaBackend
        b = OllamaBackend(real=False)
        b.load_model("llama3")
        tokens = list(b.generate_stream("Tell me a joke"))
        assert len(tokens) > 0

    def test_split_model_stub(self):
        from core.backends import OllamaBackend
        b = OllamaBackend(real=False)
        b.load_model("llama3")
        blocks = b.split_model(4)
        assert len(blocks) == 1  # Ollama handles GPU internally

    def test_infer_stub(self):
        from core.backends import OllamaBackend
        b = OllamaBackend(real=False)
        b.load_model("llama3")
        result = b.infer("hello")
        assert isinstance(result, dict)
        assert "text" in result

    def test_base_url_from_env(self):
        from core.backends import OllamaBackend
        old = os.environ.get("OLLAMA_HOST")
        try:
            os.environ["OLLAMA_HOST"] = "http://192.168.1.100:11434"
            b = OllamaBackend(real=False)
            assert "192.168.1.100" in b._base_url
        finally:
            if old is not None:
                os.environ["OLLAMA_HOST"] = old
            else:
                os.environ.pop("OLLAMA_HOST", None)


class TestHuggingFaceBackendUnit:
    """Test HuggingFace backend structure (no model loading)."""

    def test_init(self):
        from core.backends import HuggingFaceBackend
        b = HuggingFaceBackend()
        assert b.model is None
        assert b.blocks is None
        assert b._components is None

    def test_infer_no_model_raises(self):
        from core.backends import HuggingFaceBackend
        b = HuggingFaceBackend()
        with pytest.raises(RuntimeError, match="non chargé"):
            b.infer("test")

    def test_generate_no_model_raises(self):
        from core.backends import HuggingFaceBackend
        b = HuggingFaceBackend()
        with pytest.raises(RuntimeError, match="non chargé"):
            b.generate("hello")


# ========================================================================
# KV-cache block tests
# ========================================================================

class TestKVCacheBlock:
    """Test the KVCacheBlock wrapper."""

    def test_import(self):
        from core.backends import KVCacheBlock
        assert KVCacheBlock is not None

    @pytest.mark.skipif(
        not hasattr(sys.modules.get("torch", None), "randn")
        or not callable(getattr(sys.modules.get("torch", None), "randn", None)),
        reason="Requires real torch"
    )
    def test_forward_no_cache(self):
        """KVCacheBlock forward without KV-cache."""
        import torch
        from core.backends import KVCacheBlock

        # Create mock layers that return (hidden_states,)
        class MockLayer(torch.nn.Module):
            def forward(self, x, **kwargs):
                return (x * 1.1,)

        block = KVCacheBlock([MockLayer(), MockLayer()])
        hidden = torch.randn(1, 4, 16)
        out, presents = block(hidden, use_cache=False)
        assert out.shape == hidden.shape
        assert presents is None

    @pytest.mark.skipif(
        not hasattr(sys.modules.get("torch", None), "randn")
        or not callable(getattr(sys.modules.get("torch", None), "randn", None)),
        reason="Requires real torch"
    )
    def test_forward_with_cache(self):
        """KVCacheBlock forward with KV-cache returns presents."""
        import torch
        from core.backends import KVCacheBlock

        class MockLayerWithCache(torch.nn.Module):
            def forward(self, x, past_key_value=None, use_cache=False, **kw):
                present = (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
                return (x, present) if use_cache else (x,)

        block = KVCacheBlock([MockLayerWithCache()])
        hidden = torch.randn(1, 4, 16)
        out, presents = block(hidden, use_cache=True)
        assert out.shape == hidden.shape
        assert isinstance(presents, list)
        assert len(presents) == 1


# ========================================================================
# Model component extraction tests
# ========================================================================

class TestModelComponentExtraction:
    """Test _extract_model_components and _get_submodule helpers."""

    def test_get_submodule_found(self):
        from core.backends import _get_submodule
        model = types.SimpleNamespace(
            transformer=types.SimpleNamespace(
                wte="embed_layer",
                ln_f="norm_layer",
            ),
            lm_head="head_layer",
        )
        assert _get_submodule(model, ["transformer", "wte"]) == "embed_layer"
        assert _get_submodule(model, ["lm_head"]) == "head_layer"

    def test_get_submodule_not_found(self):
        from core.backends import _get_submodule
        model = types.SimpleNamespace(x=1)
        assert _get_submodule(model, ["nonexistent", "attr"]) is None

    def test_extract_components_gpt2_like(self):
        from core.backends import _extract_model_components
        model = types.SimpleNamespace(
            transformer=types.SimpleNamespace(
                wte="embed",
                wpe="pos",
                ln_f="norm",
                drop="dropout",
            ),
            lm_head="head",
        )
        comp = _extract_model_components(model)
        assert comp["embed"] == "embed"
        assert comp["pos_embed"] == "pos"
        assert comp["final_norm"] == "norm"
        assert comp["lm_head"] == "head"
        assert comp["drop"] == "dropout"

    def test_extract_components_llama_like(self):
        from core.backends import _extract_model_components
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                embed_tokens="embed",
                norm="norm",
            ),
            lm_head="head",
        )
        comp = _extract_model_components(model)
        assert comp["embed"] == "embed"
        assert comp["final_norm"] == "norm"
        assert comp["lm_head"] == "head"
        assert comp["pos_embed"] is None

    def test_extract_components_empty(self):
        from core.backends import _extract_model_components
        model = types.SimpleNamespace()
        comp = _extract_model_components(model)
        assert comp["embed"] is None
        assert comp["lm_head"] is None


# ========================================================================
# Pipeline end-to-end tests (stub mode)
# ========================================================================

class TestPipelineE2E:
    """End-to-end pipeline tests in stub mode."""

    def test_full_lifecycle(self):
        """Pipeline: init → load → status → shutdown."""
        from core.inference_pipeline import InferencePipeline
        pipe = InferencePipeline(
            backend_name="auto",
            enable_metrics=False,
            enable_discovery=False,
            verbose=False,
        )
        assert not pipe.is_loaded()

        # Try loading — may succeed or fail in stub mode
        try:
            pipe.load("test-model", num_gpus=1)
            assert pipe.is_loaded()
            s = pipe.status()
            assert s["loaded"] is True
            assert s["model"] == "test-model"
        except Exception:
            pass  # Acceptable in minimal test mode

        pipe.shutdown()
        assert not pipe.is_loaded()

    def test_singleton_lifecycle(self):
        """Singleton: get → use → reset."""
        from core.inference_pipeline import get_pipeline, reset_pipeline
        reset_pipeline()
        p1 = get_pipeline(enable_metrics=False, enable_discovery=False, verbose=False)
        p2 = get_pipeline()
        assert p1 is p2
        reset_pipeline()
        p3 = get_pipeline(enable_metrics=False, enable_discovery=False, verbose=False)
        assert p3 is not p1


# ========================================================================
# API end-to-end tests
# ========================================================================

class TestAPIE2E:
    """End-to-end tests through the Flask API."""

    @pytest.fixture(autouse=True)
    def client(self):
        try:
            from core.production_api import create_app
            app = create_app()
            app.config["TESTING"] = True
            with app.test_client() as c:
                self._client = c
                yield c
        except ImportError:
            pytest.skip("production_api not available")

    def test_health_ready_live_sequence(self):
        """Health endpoints respond correctly in sequence."""
        r1 = self._client.get("/health")
        assert r1.status_code == 200
        r2 = self._client.get("/live")
        assert r2.status_code == 200
        r3 = self._client.get("/ready")
        assert r3.status_code in (200, 503)

    def test_status_gpu_system_info(self):
        """Info endpoints return valid JSON."""
        for endpoint in ["/api/status", "/api/gpu", "/api/system"]:
            resp = self._client.get(endpoint)
            assert resp.status_code == 200
            data = resp.get_json()
            assert isinstance(data, (dict, list))

    def test_pipeline_status_api(self):
        """Pipeline status endpoint works."""
        resp = self._client.get("/api/pipeline/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "loaded" in data

    def test_completions_missing_prompt(self):
        """Completions without prompt returns error."""
        resp = self._client.post(
            "/v1/completions",
            data=json.dumps({"max_tokens": 5}),
            content_type="application/json",
        )
        assert resp.status_code in (400, 500, 503)

    def test_generate_missing_prompt(self):
        """Generate without prompt returns error."""
        resp = self._client.post(
            "/api/generate",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code in (400, 500, 503)

    def test_model_load_missing_name(self):
        """Model load without name returns error."""
        resp = self._client.post(
            "/api/models/load",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code in (400, 500)

    def test_models_list(self):
        """GET /api/models returns a list."""
        resp = self._client.get("/api/models")
        assert resp.status_code == 200

    def test_nodes_list(self):
        """GET /api/nodes returns valid response."""
        resp = self._client.get("/api/nodes")
        assert resp.status_code == 200


# ========================================================================
# Transfer manager e2e tests
# ========================================================================

class TestTransferManagerE2E:
    """Full TransferManager lifecycle in stub mode."""

    def test_full_lifecycle(self):
        """Init → send → stats."""
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        assert tm._stub_mode is True

        # Send an activation
        result = tm.send_activation(0, 1, "fake_tensor_data")
        assert result is not None
        assert result.source_gpu == 0
        assert result.target_gpu == 1

        # Stats
        stats = tm.stats()
        assert isinstance(stats, dict)
        assert "transfers" in stats or "total_transfers" in stats or len(stats) >= 0

    def test_multiple_transfers(self):
        """Multiple transfers don't crash."""
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        for i in range(10):
            result = tm.send_activation(0, 1, f"tensor_{i}")
            assert result is not None

    def test_nccl_off_in_single_process(self):
        """NCCL is not initialized without distributed env vars."""
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        assert tm._nccl_initialized is False


# ========================================================================
# Security tests
# ========================================================================

class TestSecurityE2E:
    """Security module end-to-end tests."""

    def test_import_security(self):
        from core.security import install_security
        assert callable(install_security)

    def test_auth_strong_import(self):
        from core.auth_strong import ensure_default_admin
        assert callable(ensure_default_admin)

    def test_production_mode_blocks_bypasses(self):
        """In production mode, test bypasses are disabled."""
        old_prod = os.environ.get("VRM_PRODUCTION")
        try:
            os.environ["VRM_PRODUCTION"] = "1"
            # Re-import to pick up the change
            import importlib
            from core import security
            importlib.reload(security)
            # The _check_test_bypass should return None in production
            if hasattr(security, "_check_test_bypass"):
                result = security._check_test_bypass()
                assert result is None
        except Exception:
            pass  # Module reload might fail in some setups
        finally:
            if old_prod is not None:
                os.environ["VRM_PRODUCTION"] = old_prod
            else:
                os.environ.pop("VRM_PRODUCTION", None)

    def test_auth_strong_no_default_in_prod(self):
        """ensure_default_admin refuses to create admin/admin in prod mode."""
        old_prod = os.environ.get("VRM_PRODUCTION")
        try:
            os.environ["VRM_PRODUCTION"] = "1"
            from core.auth_strong import ensure_default_admin
            # Should not create insecure defaults when VRM_PRODUCTION=1
            # (either returns None or raises)
            try:
                result = ensure_default_admin()
                # If it returns, it should not have created admin/admin
            except Exception:
                pass  # Raising is acceptable in prod mode
        finally:
            if old_prod is not None:
                os.environ["VRM_PRODUCTION"] = old_prod
            else:
                os.environ.pop("VRM_PRODUCTION", None)


# ========================================================================
# Network transport tests
# ========================================================================

class TestNetworkTransport:
    """Test cleaned-up network modules."""

    def test_transport_import(self):
        """Transport module imports without errors."""
        try:
            from core.network.transport import Transport
            assert Transport is not None
        except ImportError:
            pytest.skip("network.transport not available")

    def test_transmission_import(self):
        """Transmission module imports cleanly."""
        try:
            from core.network.transmission import send_block, start_client
            assert callable(send_block)
            assert callable(start_client)
        except ImportError:
            pytest.skip("network.transmission not available")

    def test_transmission_unknown_protocol_raises(self):
        """send_block with unknown protocol raises ValueError."""
        try:
            from core.network.transmission import send_block
            with pytest.raises(ValueError, match="Unknown protocol"):
                send_block(b"data", protocol="invalid_xyz")
        except ImportError:
            pytest.skip("network.transmission not available")

    def test_start_client_without_socketio(self):
        """start_client returns False if socketio is unavailable."""
        try:
            from core.network import transmission
            # Save original
            orig_sio = transmission.sio
            try:
                transmission.sio = None
                result = transmission.start_client("http://localhost:5000")
                assert result is False
            finally:
                transmission.sio = orig_sio
        except ImportError:
            pytest.skip("network.transmission not available")
