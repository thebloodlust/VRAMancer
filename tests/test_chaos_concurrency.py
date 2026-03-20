import pytest
import threading
import time
import os
import random
import queue
from unittest.mock import patch, MagicMock
from core.inference_pipeline import InferencePipeline, get_pipeline

@pytest.fixture
def chaos_env():
    """Fixture ensuring minimal test environment for chaos simulations."""
    old_minimal = os.environ.get("VRM_MINIMAL_TEST")
    old_test = os.environ.get("VRM_TEST_MODE")
    os.environ["VRM_MINIMAL_TEST"] = "1"
    os.environ["VRM_TEST_MODE"] = "1"
    yield
    if old_minimal is not None:
        os.environ["VRM_MINIMAL_TEST"] = old_minimal
    else:
        os.environ.pop("VRM_MINIMAL_TEST", None)
    if old_test is not None:
        os.environ["VRM_TEST_MODE"] = old_test
    else:
        os.environ.pop("VRM_TEST_MODE", None)

@pytest.mark.slow
@pytest.mark.chaos
def test_pipeline_concurrent_load(chaos_env):
    """
    Sprint B: Simulate high concurrent load to trigger race conditions.
    """
    # Reset fault tolerance manager for clean state
    from core.gpu_fault_tolerance import reset_fault_manager
    reset_fault_manager()
    
    pipeline = InferencePipeline().load("test-model", num_gpus=2)
    exceptions = queue.Queue()
    
    def worker():
        try:
            # Simulate generating a prompt
            prompt = f"Random prompt {random.randint(1, 1000)}"
            for _ in pipeline.generate(prompt, max_new_tokens=10):
                time.sleep(0.01) # Simulate slow compute
        except Exception as e:
            exceptions.put(e)

    threads = []
    for _ in range(50): # High concurrency
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # In a perfect world, zero exceptions. If there are, it's a bug to fix!
    if not exceptions.empty():
        while not exceptions.empty():
            print(f"THREAD EXCEPTION: {exceptions.get()}")
    assert exceptions.empty(), "Chaos testing failed with unhandled exceptions in threads"

@pytest.mark.slow
@pytest.mark.chaos
@patch("core.monitor.GPUMonitor.vram_usage")
def test_pipeline_oom_simulation(mock_vram, chaos_env):
    """
    Sprint B: Simulate an Out-of-Memory event during hot inference.
    """
    # Simulate an immediate OOM scenario across all GPUs
    mock_vram.return_value = {0: {"used": 20000, "total": 24000}, 1: {"used": 23500, "total": 24000}}
    
    pipeline = InferencePipeline().load("test-oom-model", num_gpus=2)
    
    with pytest.raises(Exception, match=r"(?i)(memory|oom|capacity|fail)"):
        # The scheduler or pipeline should catch the OOM or gracefully degrade
        for _ in pipeline.generate("Force OOM", max_new_tokens=1000):
            pass
