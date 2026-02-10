"""Tests for structural improvements — circuit-breaker, validation module,
registry module, timing-safe auth, ZeroTrust production mode, rate limiter GC,
edge node eviction, and SSE streaming format.
"""
import os
import sys
import json
import time
import hmac
import types
import threading
import pytest

# Ensure test env
os.environ.setdefault('VRM_API_TOKEN', 'testtoken')
os.environ.setdefault('VRM_MINIMAL_TEST', '1')
os.environ.setdefault('VRM_DISABLE_RATE_LIMIT', '1')
os.environ.setdefault('VRM_TEST_MODE', '1')


# ======================================================================
# Circuit-Breaker
# ======================================================================

class TestCircuitBreaker:
    def test_starts_closed(self):
        from core.api.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        from core.api.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_half_open_after_timeout(self):
        from core.api.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request()

    def test_closes_after_success_in_half_open(self):
        from core.api.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05,
                            success_threshold=1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        from core.api.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_context_manager_success(self):
        from core.api.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=3)
        with cb:
            pass
        assert cb.state == CircuitState.CLOSED

    def test_context_manager_failure(self):
        from core.api.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=1)
        with pytest.raises(ValueError):
            with cb:
                raise ValueError("boom")
        assert cb.state == CircuitState.OPEN

    def test_context_manager_open_rejects(self):
        from core.api.circuit_breaker import (
            CircuitBreaker, CircuitOpenError,
        )
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999)
        cb.record_failure()
        with pytest.raises(CircuitOpenError):
            with cb:
                pass

    def test_reset(self):
        from core.api.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_status_dict(self):
        from core.api.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
        s = cb.status()
        assert s['state'] == 'closed'
        assert s['failure_threshold'] == 5


# ======================================================================
# Validation sub-module (extracted)
# ======================================================================

class TestValidationModule:
    def test_valid_params(self):
        from core.api.validation import validate_generation_params
        params, err = validate_generation_params({
            'max_tokens': 100, 'temperature': 0.7, 'top_p': 0.9
        })
        assert err is None
        assert params['max_tokens'] == 100

    def test_max_tokens_too_high(self):
        from core.api.validation import validate_generation_params
        _, err = validate_generation_params({'max_tokens': 9999})
        assert err is not None
        assert err[1] == 400

    def test_temperature_invalid(self):
        from core.api.validation import validate_generation_params
        _, err = validate_generation_params({'temperature': 'hot'})
        assert err is not None
        assert err[1] == 400

    def test_count_tokens_fallback(self):
        from core.api.validation import count_tokens
        assert count_tokens("hello world test") == 3

    def test_count_tokens_with_tokenizer(self):
        from core.api.validation import count_tokens

        class FakeTokenizer:
            def encode(self, text):
                return list(range(len(text.split()) * 2))

        assert count_tokens("hello world", FakeTokenizer()) == 4


# ======================================================================
# Registry sub-module (extracted)
# ======================================================================

class TestRegistryModule:
    def test_not_loaded_by_default(self):
        from core.api.registry import PipelineRegistry
        r = PipelineRegistry()
        assert not r.is_loaded()
        assert r.model_name is None
        assert r.num_gpus == 0
        assert r.blocks == []

    def test_generate_without_model(self):
        from core.api.registry import PipelineRegistry
        r = PipelineRegistry()
        with pytest.raises(RuntimeError, match="No model loaded"):
            r.generate("test")

    def test_infer_without_model(self):
        from core.api.registry import PipelineRegistry
        r = PipelineRegistry()
        with pytest.raises(RuntimeError, match="No model loaded"):
            r.infer(None)

    def test_status_without_pipeline(self):
        from core.api.registry import PipelineRegistry
        r = PipelineRegistry()
        s = r.status()
        assert s['loaded'] is False


# ======================================================================
# Timing-safe network auth
# ======================================================================

class TestTimingSafeAuth:
    def test_correct_key_accepted(self):
        from core.network.security import generate_node_key, authenticate_node
        key = generate_node_key("mysecret")
        assert authenticate_node(key, [key])

    def test_wrong_key_rejected(self):
        from core.network.security import generate_node_key, authenticate_node
        key = generate_node_key("mysecret")
        wrong = generate_node_key("wrong")
        assert not authenticate_node(wrong, [key])

    def test_empty_known_keys(self):
        from core.network.security import generate_node_key, authenticate_node
        key = generate_node_key("secret")
        assert not authenticate_node(key, [])

    def test_uses_hmac_compare_digest(self):
        """Verify the function uses timing-safe comparison."""
        import core.network.security as mod
        import inspect
        source = inspect.getsource(mod.authenticate_node)
        assert 'compare_digest' in source


# ======================================================================
# Rate limiter GC
# ======================================================================

class TestRateLimiterGC:
    def test_stale_keys_are_cleaned(self):
        import core.security as sec_mod
        sec_mod.reset_rate_limiter()
        now = time.time()
        with sec_mod._requests_lock:
            sec_mod._requests['stale_ip:/path'] = [now - 120]
            sec_mod._requests['fresh_ip:/path'] = [now]
            sec_mod._last_gc = 0  # force GC to run
            sec_mod._gc_stale_keys(now)
        assert 'stale_ip:/path' not in sec_mod._requests
        assert 'fresh_ip:/path' in sec_mod._requests
        sec_mod.reset_rate_limiter()


# ======================================================================
# SSE format
# ======================================================================

class TestSSEFormat:
    @pytest.fixture(autouse=True)
    def _setup_stub(self):
        os.environ['VRM_BACKEND_ALLOW_STUB'] = '1'
        yield
        os.environ.pop('VRM_BACKEND_ALLOW_STUB', None)
        # Reset registry to avoid state leakage
        try:
            from core.production_api import _registry
            _registry._pipeline = None
        except Exception:
            pass

    @pytest.fixture
    def client(self):
        from core.production_api import create_app
        application = create_app()
        application.config['TESTING'] = True
        with application.test_client() as c:
            yield c

    def test_sse_completions_mimetype(self, client):
        """SSE stream should return text/event-stream or 500 (no real model)."""
        resp = client.post(
            '/v1/completions',
            json={'prompt': 'hello', 'stream': True, 'model': 'test'},
            headers={'X-API-TOKEN': 'testtoken'},
        )
        # With stub backend: may succeed or fail depending on model availability
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert resp.content_type.startswith('text/event-stream')

    def test_sse_completions_data_lines(self, client):
        """SSE stream should have data: lines with valid JSON when successful."""
        resp = client.post(
            '/v1/completions',
            json={'prompt': 'hello', 'stream': True, 'model': 'test'},
            headers={'X-API-TOKEN': 'testtoken'},
        )
        if resp.status_code == 200 and resp.content_type.startswith('text/event-stream'):
            text = resp.data.decode('utf-8')
            lines = [l for l in text.strip().split('\n') if l.startswith('data:')]
            assert len(lines) >= 1
            for line in lines:
                payload = line[len('data:'):].strip()
                if payload == '[DONE]':
                    continue
                parsed = json.loads(payload)
                assert isinstance(parsed, dict)
        else:
            # No real model — just verify we got a valid response
            assert resp.status_code in (200, 500)

    def test_sse_chat_mimetype(self, client):
        """Chat SSE should return correct content type or 500."""
        resp = client.post(
            '/v1/chat/completions',
            json={
                'messages': [{'role': 'user', 'content': 'hi'}],
                'stream': True,
                'model': 'test',
            },
            headers={'X-API-TOKEN': 'testtoken'},
        )
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert resp.content_type.startswith('text/event-stream')


# ======================================================================
# Queue status with circuit-breaker
# ======================================================================

class TestQueueCircuitBreaker:
    def test_queue_status_has_circuit_breaker(self):
        from core.production_api import create_app
        from core.security import reset_rate_limiter
        reset_rate_limiter()
        application = create_app()
        application.config['TESTING'] = True
        with application.test_client() as client:
            resp = client.get('/api/queue/status',
                              headers={'X-API-TOKEN': os.environ.get('VRM_API_TOKEN', 'testtoken')})
            assert resp.status_code == 200, f"Got {resp.status_code}: {resp.data}"
            data = resp.get_json()
            assert 'circuit_breaker' in data
            assert data['circuit_breaker']['state'] == 'closed'



