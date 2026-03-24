"""
Backward-compatibility redirect — use ``core.parity_memory`` instead.

This module was renamed from ``holographic_memory`` to ``parity_memory``
for honest naming (it's XOR parity, not holographic storage).
"""
from core.parity_memory import ParityKVManager as HolographicKVManager  # noqa: F401
from core.parity_memory import parity_kv as hive_memory  # noqa: F401
