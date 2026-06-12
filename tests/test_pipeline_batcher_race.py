"""Race condition test for continuous_batcher start/submit lifecycle."""
import os
import threading
import time

import pytest


@pytest.mark.skipif(
    not os.environ.get("VRM_MINIMAL_TEST"),
    reason="Requires stub-safe environment",
)
def test_batcher_concurrent_submit_after_load(monkeypatch):
    monkeypatch.setenv("VRM_CONTINUOUS_BATCHING", "1")

    try:
        from core.inference_pipeline import InferencePipeline
    except ImportError:
        pytest.skip("InferencePipeline unavailable")

    pipe = InferencePipeline(
        backend_name="huggingface",
        enable_metrics=False,
        enable_discovery=False,
        verbose=False,
    )
    try:
        pipe.load("gpt2", num_gpus=1)
    except Exception:
        pytest.skip("Cannot load model in minimal-test mode")

    if pipe.continuous_batcher is None:
        pytest.skip("Batcher disabled in this build")

    errors = []

    def worker():
        try:
            fut = pipe.continuous_batcher.submit(
                "hi", max_new_tokens=1, temperature=1.0,
                top_k=1, top_p=1.0,
            )
            assert fut is not None
        except Exception as exc:
            errors.append(exc)

    pipe.continuous_batcher.start()
    time.sleep(0.01)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    try:
        pipe.shutdown()
    except Exception:
        pass

    assert not errors, f"Race surfaced: {errors[:3]}"
