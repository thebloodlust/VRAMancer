"""T7.6 — Auto-heal du serveur d'inférence (Phase 7).

Validates the OOM recovery ladder added to ``InferencePipeline``:

  (a)/(b) generation-time OOM -> cache cleared, lending leases reclaimed,
          retried with reduced tokens, AND the reduction persists for
          every subsequent request (``_max_new_tokens_scale``).
  (c)/(d) load-time OOM -> retried with +10% VRAM reserve margin, and as
          a last resort reloaded in NF4 (pipeline marked ``degraded``).

Also checks that ``/health`` (via ``PipelineRegistry.health_extra()``)
surfaces the degraded state.

These tests stub out the heavy backend calls (no real model load / no
real OOM) — they exercise the recovery *logic* in
``core/inference_pipeline.py``, not CUDA itself.
"""
import os

import pytest

from core.inference_pipeline import InferencePipeline


@pytest.fixture
def minimal_env():
    old = os.environ.get("VRM_MINIMAL_TEST")
    os.environ["VRM_MINIMAL_TEST"] = "1"
    yield
    if old is not None:
        os.environ["VRM_MINIMAL_TEST"] = old
    else:
        os.environ.pop("VRM_MINIMAL_TEST", None)


def _new_pipeline(minimal_env):
    return InferencePipeline(enable_metrics=False).load("test-model", num_gpus=1)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="_oom_recover() short-circuits without CUDA",
)
def test_generation_oom_recovers_and_persists_degradation(minimal_env):
    """A single generation-time OOM is survived AND degrades future requests."""
    pipeline = _new_pipeline(minimal_env)
    assert pipeline.degraded is False
    assert pipeline._max_new_tokens_scale == 1.0

    calls = {"n": 0}

    def flaky_generate(prompt, gen_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return f"ok max_new_tokens={gen_kwargs.get('max_new_tokens')}"

    pipeline._protected_generate = flaky_generate

    # (a) the request itself succeeds despite the OOM
    out = pipeline.generate("hello", max_new_tokens=128)
    assert out.startswith("ok")

    # (b) degradation is recorded and persists
    assert pipeline.degraded is True
    assert pipeline.degraded_reason == "oom_context_reduced"
    assert pipeline._max_new_tokens_scale == pytest.approx(0.75)

    # A later request gets its max_new_tokens capped by the persisted scale,
    # WITHOUT needing another OOM.
    captured = {}

    def capturing_generate(prompt, gen_kwargs):
        captured["max_new_tokens"] = gen_kwargs.get("max_new_tokens")
        return "ok"

    pipeline._protected_generate = capturing_generate
    pipeline.generate("hello again", max_new_tokens=128)
    assert captured["max_new_tokens"] == 96  # 128 * 0.75


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="_oom_recover() short-circuits without CUDA",
)
def test_repeated_oom_recovery_is_stable(minimal_env):
    """5 consecutive OOM-recovery cycles: process survives, scale floors at 0.1."""
    pipeline = _new_pipeline(minimal_env)

    for _ in range(5):
        calls = {"n": 0}

        def flaky_generate(prompt, gen_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("CUDA out of memory.")
            return "ok"

        pipeline._protected_generate = flaky_generate
        out = pipeline.generate("prompt", max_new_tokens=128)
        assert out == "ok"

    assert pipeline.degraded is True
    assert pipeline.degraded_reason == "oom_context_reduced"
    # 0.75^5 ~= 0.237, floored at 0.1
    assert 0.1 <= pipeline._max_new_tokens_scale < 1.0


def test_load_time_oom_retries_with_extra_margin_then_nf4(minimal_env, monkeypatch):
    """Load-time OOM ladder: (c) +10% margin retry, then (d) NF4 fallback."""
    pipeline = InferencePipeline(enable_metrics=False)

    calls = {"n": 0}
    seen_extra_reserve = []
    seen_quant_env = []

    class FakeBackend:
        backend_type = "huggingface"
        lending_pool = None
        _extra_load_reserve = 0.0

        def load_model(self, model_name, **kwargs):
            calls["n"] += 1
            seen_extra_reserve.append(self._extra_load_reserve)
            seen_quant_env.append(os.environ.get("VRM_QUANTIZATION", ""))
            if calls["n"] < 3:
                raise RuntimeError("CUDA out of memory. Tried to allocate 10.00 GiB")
            # 3rd attempt (NF4 fallback) succeeds
            self.model_name = model_name
            self.model = "STUB_MODEL"
            self.tokenizer = "STUB_TOKENIZER"

    fake_backend = FakeBackend()
    monkeypatch.setattr("core.backends.select_backend", lambda *a, **k: fake_backend)

    pipeline.load("test-oom-load-model", num_gpus=1)

    assert calls["n"] == 3
    # Attempt 2 used the +10% margin
    assert seen_extra_reserve[1] == pytest.approx(0.10)
    # Attempt 3 (NF4 fallback) had VRM_QUANTIZATION=nf4 set
    assert seen_quant_env[2] == "nf4"

    assert pipeline.degraded is True
    assert pipeline.degraded_reason == "oom_fallback_nf4"
    os.environ.pop("VRM_QUANTIZATION", None)


def test_health_extra_surfaces_degraded_state(minimal_env):
    from core.api.registry import PipelineRegistry

    registry = PipelineRegistry()
    assert registry.health_extra() == {"degraded": False, "degraded_reason": None}

    pipeline = _new_pipeline(minimal_env)
    pipeline.degraded = True
    pipeline.degraded_reason = "oom_fallback_nf4"
    registry._pipeline = pipeline

    extra = registry.health_extra()
    assert extra["degraded"] is True
    assert extra["degraded_reason"] == "oom_fallback_nf4"
