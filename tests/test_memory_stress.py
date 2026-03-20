# tests/test_memory_stress.py

import os
import pytest
import torch
from core.memory_monitor import get_ram_status, is_ram_saturated

@pytest.mark.slow
def simulate_memory_stress():
    if not os.environ.get("ENABLE_STRESS_TEST"):
        pytest.skip("Stress test désactivé (export ENABLE_STRESS_TEST=1 pour activer)")
    print(" RAM avant allocation :", get_ram_status())
    tensors = []
    try:
        for i in range(4):  # réduit pour éviter OOM
            t = torch.randn(384, 384, 384)
            tensors.append(t)
            print(f" Allocation {i+1} → RAM saturée ? {is_ram_saturated()}")
    except RuntimeError as e:
        print(" Erreur d’allocation :", e)
    print(" RAM après allocation :", get_ram_status())

if __name__ == "__main__":
    simulate_memory_stress()
