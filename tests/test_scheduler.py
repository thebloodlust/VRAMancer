import pytest
from core.scheduler import Scheduler
from core.memory_block import MemoryBlock

# Mock GPU list
mock_gpus = [
    {"id": 0, "name": "GPU-A", "total_vram_mb": 2048, "is_available": True},
    {"id": 1, "name": "GPU-B", "total_vram_mb": 1024, "is_available": True}
]

def test_scheduler_initialization():
    sched = Scheduler(gpu_list=mock_gpus)
    assert sched.strategy == "balanced"
    assert len(sched.allocations) == 2

def test_block_allocation():
    sched = Scheduler(gpu_list=mock_gpus)
    block = sched.allocate_block(512)
    assert isinstance(block, MemoryBlock)
    assert block.size_mb == 512
    assert block.status == "allocated"

def test_allocation_overflow():
    sched = Scheduler(gpu_list=mock_gpus)
    with pytest.raises(RuntimeError):
        sched.allocate_block(99999)  # Trop gros pour les GPU mockés
