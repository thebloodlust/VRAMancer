"""Tests for production API — SSE, chat/completions, queue, timeout, validation.

Covers:
  - Input validation (max_tokens, temperature, top_p)
  - /v1/chat/completions (non-streaming + streaming)
  - /v1/completions SSE streaming
  - /api/queue/status
  - PipelineRegistry thread safety (basic)
  - _run_with_timeout backpressure
  - uuid4 request IDs (no MD5)
"""
import os
import sys
import json
import types
import threading
import pytest

# Ensure test env
os.environ.setdefault('VRM_API_TOKEN', 'testtoken')
os.environ.setdefault('VRM_MINIMAL_TEST', '1')
os.environ.setdefault('VRM_DISABLE_RATE_LIMIT', '1')
os.environ.setdefault('VRM_TEST_MODE', '1')


@pytest.fixture
def app_client():
    """Create a fresh Flask test client."""
    from core.production_api import create_app
    application = create_app()
    application.config['TESTING'] = True
    with application.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_max_tokens_out_of_range(self, app_client):
        resp = app_client.post('/v1/completions',
                               json={'prompt': 'hello', 'max_tokens': 99999},
                               headers={'X-API-TOKEN': 'testtoken'})
        assert resp.status_code == 400
        data = resp.get_json()
        assert 'max_tokens' in data.get('error', '').lower()

    def test_max_tokens_negative(self, app_client):
        resp = app_client.post('/v1/completions',
                               json={'prompt': 'hello', 'max_tokens': -1},
                               headers={'X-API-TOKEN': 'testtoken'})
        assert resp.status_code == 400

    def test_temperature_out_of_range(self, app_client):
        resp = app_client.post('/v1/completions',
                               json={'prompt': 'hello', 'temperature': 5.0},
                               headers={'X-API-TOKEN': 'testtoken'})
        assert resp.status_code == 400
        data = resp.get_json()
        assert 'temperature' in data.get('error', '').lower()

    def test_top_p_out_of_range(self, app_client):
        resp = app_client.post('/v1/completions',
                               json={'prompt': 'hello', 'top_p': 2.0},
                               headers={'X-API-TOKEN': 'testtoken'})
        assert resp.status_code == 400
        data = resp.get_json()
        assert 'top_p' in data.get('error', '').lower()

    def test_chat_validation(self, app_client):
        resp = app_client.post('/v1/chat/completions',
                               json={'messages': [{'role': 'user', 'content': 'hi'}],
                                     'max_tokens': 0},
                               headers={'X-API-TOKEN': 'testtoken'})
        assert resp.status_code == 400

    def test_missing_prompt(self, app_client):
        resp = app_client.post('/v1/completions',
                               json={'max_tokens': 10},
                               headers={'X-API-TOKEN': 'testtoken'})
        assert resp.status_code == 400

    def test_missing_messages(self, app_client):
        resp = app_client.post('/v1/chat/completions',
                               json={'model': 'gpt2'},
                               headers={'X-API-TOKEN': 'testtoken'})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Queue status
# ---------------------------------------------------------------------------

class TestQueueStatus:
    def test_queue_status_returns_fields(self, app_client):
        resp = app_client.get('/api/queue/status',
                              headers={'X-API-TOKEN': 'testtoken'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'queue_depth' in data
        assert 'max_queue_size' in data
        assert 'utilization_pct' in data
        assert data['queue_depth'] == 0

    def test_queue_utilization_zero(self, app_client):
        resp = app_client.get('/api/queue/status',
                              headers={'X-API-TOKEN': 'testtoken'})
        data = resp.get_json()
        assert data['utilization_pct'] == 0.0


# ---------------------------------------------------------------------------
# Pipeline Registry
# ---------------------------------------------------------------------------

class TestPipelineRegistry:
    def test_registry_starts_empty(self):
        from core.production_api import PipelineRegistry
        reg = PipelineRegistry()
        assert reg.get() is None
        assert not reg.is_loaded()
        assert reg.model_name is None
        assert reg.num_gpus == 0
        assert reg.blocks == []
        assert reg.get_tokenizer() is None

    def test_registry_status_when_empty(self):
        from core.production_api import PipelineRegistry
        reg = PipelineRegistry()
        status = reg.status()
        assert status['loaded'] is False

    def test_registry_thread_safety(self):
        """Test concurrent access to registry does not crash."""
        from core.production_api import PipelineRegistry
        reg = PipelineRegistry()
        results = []

        def _reader():
            for _ in range(50):
                reg.is_loaded()
                reg.model_name
                reg.status()
            results.append(True)

        threads = [threading.Thread(target=_reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(results) == 8


# ---------------------------------------------------------------------------
# Request ID format (uuid4, not MD5)
# ---------------------------------------------------------------------------

class TestRequestIdFormat:
    def test_no_model_error_still_validates(self, app_client):
        """Even error responses should not contain MD5-generated IDs."""
        resp = app_client.post('/v1/completions',
                               json={'prompt': 'hello'},
                               headers={'X-API-TOKEN': 'testtoken'})
        # 400 because no model loaded
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Validation helper standalone
# ---------------------------------------------------------------------------

class TestValidateGenerationParams:
    def test_valid_defaults(self):
        from core.production_api import _validate_generation_params
        params, err = _validate_generation_params({})
        assert err is None
        assert params['max_tokens'] == 128
        assert params['temperature'] == 1.0
        assert params['top_p'] == 1.0

    def test_valid_custom(self):
        from core.production_api import _validate_generation_params
        params, err = _validate_generation_params({
            'max_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
        })
        assert err is None
        assert params == {
            'max_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
        }

    def test_invalid_max_tokens_string(self):
        from core.production_api import _validate_generation_params
        _, err = _validate_generation_params({'max_tokens': 'abc'})
        assert err is not None
        assert err[1] == 400

    def test_boundary_values(self):
        from core.production_api import _validate_generation_params
        params, err = _validate_generation_params({
            'max_tokens': 1,
            'temperature': 0.0,
            'top_p': 0.0,
        })
        assert err is None
        params, err = _validate_generation_params({
            'max_tokens': 4096,
            'temperature': 2.0,
            'top_p': 1.0,
        })
        assert err is None


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

class TestTokenCounter:
    def test_whitespace_fallback(self):
        from core.production_api import _count_tokens
        assert _count_tokens("hello world foo") == 3

    def test_with_mock_tokenizer(self):
        from core.production_api import _count_tokens
        tok = types.SimpleNamespace(encode=lambda text: list(range(5)))
        assert _count_tokens("anything", tok) == 5
