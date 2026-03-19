import pytest
import threading
import time
import os
import random
from unittest.mock import patch, MagicMock
from core.inference_pipeline import InferencePipeline

@pytest.fixture
def chaos_env():
    """Fixture ensuring minimal test environment for chaos simulations."""
    os.environ["VRM_MINIMAL_TEST"] = "1"
    os.environ["VRM_TEST_MODE"] = "1"
    yield
    os.environ.pop("VRM_MINIMAL_TEST", None)
    os.environ.pop("VRM_TEST_MODE", None)

def test_pipeline_concurrent_load(chaos_env):
    """
    Sprint B: Simulate high concurrent load to trigger race conditions.
    """
    pipeline = InferencePipeline.load("test-model", num_gpus=2)
    exceptions = []
    
    def worker():
        try:
            # Simulate generating a prompt
            prompt = f"Random prompt {random.randint(1, 1000)}"
            for _ in pipeline.generate(prompt, max_new_tokens=10):
                time.sleep(0.01) # Simulate slow compute
        except Exception as e:
            exceptions.append(e)

    threads = []
    for _ in range(50): # High concurrency
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # In a perfect world, zero exceptions. If there are, it's a bug to fix!
    assert len(exceptions) == 0, f"Chaos testing failed with {len(exceptions)} unhandled exceptions in threads: {exceptions}"

@patch("core.monitor.GPUMonitor.vram_usage")
def test_pipeline_oom_simulation(mock_vram, chaos_env):
    """
    Sprint B: Simulate an Out-of-Memory event during hot inference.
    """
    # Simulate an immediate OOM scenario across all GPUs
    mock_vram.return_value = {0: {"used": 20000, "total": 24000}, 1: {"used": 23500, "total": 24000}}
    
    pipeline = InferencePipeline.load("test-oom-model", num_gpus=2)
    
    with pytest.raises(Exception) as exc_info:
        # The scheduler or pipeline should catch the OOM or gracefully degrade
        try:
            for _ in pipeline.generate("Force OOM", max_new_tokens=1000):
                pass
        except Exception as e:
            # Re-raise to match pytest.raises
            raise e
            
    # Success means the system didn't hang, it explicitly crashed or handled it
    assert "Memory" in str(exc_info.value) or "OOM" in str(exc_info.value) or "capacity" in str(exc_info.value).lower(), \
        f"Expected an OOM/Memory error but got: {exc_info.value}"
