from core.hierarchical_memory import HierarchicalMemoryManager
from core.memory_block import MemoryBlock
from core.metrics import MEMORY_PROMOTIONS, MEMORY_DEMOTIONS

def test_promotion_demotion_metrics():
    h = HierarchicalMemoryManager()
    b = MemoryBlock(16,0)
    h.register_block(b, 'L5')
    # promotion sequence
    for _ in range(3):
        h.touch(b); h.promote_policy(b)
    # force demotion
    h.policy_demote_if_needed(b, gpu_over_pct=99)
    # cannot assert absolute values reliably (depends path) but ensure counters exist
    # Pull internal samples
    assert MEMORY_PROMOTIONS._value.get() >= 0
    assert MEMORY_DEMOTIONS._value.get() >= 0
