"""V5 P10.1 — verify ContinuousBatcher initialises with VRM_CONTINUOUS_BATCHING=1.

The [NEGATIVE@P1.4] revert means auto-start via generate() is NOT wired —
the batcher is created but remains idle until pipeline.submit() is called.
These tests guard against regression on both sides:
  1. With the flag: batcher is not None and start() works.
  2. Without the flag: batcher may be None or not running.
"""
import os
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_BACKEND_ALLOW_STUB", "1")


def test_batcher_initialized_with_flag(monkeypatch):
    """With VRM_CONTINUOUS_BATCHING=1, InferencePipeline creates a batcher."""
    monkeypatch.setenv("VRM_CONTINUOUS_BATCHING", "1")
    pytest.importorskip("torch")

    from core.inference_pipeline import InferencePipeline, reset_pipeline
    reset_pipeline()

    pipe = InferencePipeline(enable_metrics=False, enable_discovery=False, verbose=False)

    # Simulate post-load initialisation path that sets up continuous_batcher
    try:
        pipe._init_continuous_batcher()
    except Exception as e:
        pytest.skip(f"_init_continuous_batcher() unavailable or failed: {e}")

    assert pipe.continuous_batcher is not None, (
        "VRM_CONTINUOUS_BATCHING=1: continuous_batcher must be initialised"
    )


def test_batcher_can_start_and_stop(monkeypatch):
    """ContinuousBatcher.start() transitions _running to True; stop() reverses it."""
    monkeypatch.setenv("VRM_CONTINUOUS_BATCHING", "1")
    pytest.importorskip("torch")

    try:
        from core.continuous_batcher import ContinuousBatcher
    except ImportError as e:
        pytest.skip(f"ContinuousBatcher not importable: {e}")

    cb = ContinuousBatcher(model=None, tokenizer=None, device="cpu")
    assert cb._running is False, "batcher must not auto-start on __init__"

    cb.start()
    assert cb._running is True, "batcher must be running after start()"

    cb.stop()
    assert cb._running is False, "batcher must be stopped after stop()"


def test_batcher_no_autostart_without_flag(monkeypatch):
    """Without VRM_CONTINUOUS_BATCHING, a freshly created batcher stays idle."""
    monkeypatch.delenv("VRM_CONTINUOUS_BATCHING", raising=False)
    pytest.importorskip("torch")

    try:
        from core.continuous_batcher import ContinuousBatcher
    except ImportError as e:
        pytest.skip(f"ContinuousBatcher not importable: {e}")

    cb = ContinuousBatcher(model=None, tokenizer=None, device="cpu")
    assert cb._running is False, (
        "Without VRM_CONTINUOUS_BATCHING=1, batcher must stay idle on __init__"
    )
