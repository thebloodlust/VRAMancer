import os
import pytest
from core.hierarchical_memory import HierarchicalMemoryManager
from core.memory_block import MemoryBlock

def test_promotion_cycle():
    h = HierarchicalMemoryManager()
    b = MemoryBlock(size_mb=32, gpu_id=0, status="allocated")
    h.register_block(b, "L5")
    # Simule accès répétés
    for i in range(9):
        h.touch(b)
        h.promote_policy(b)
    tier = h.get_tier(b.id)
    # Après assez d'accès on doit être remonté au moins à L3 ou mieux
    assert tier in {"L3","L2","L1"}
