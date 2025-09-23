# tests/test_memory_stress.py

import torch
from core.memory_monitor import get_ram_status, is_ram_saturated

def simulate_memory_stress():
    print("📊 RAM avant allocation :", get_ram_status())
    tensors = []
    try:
        for i in range(20):
            t = torch.randn(512, 512, 512)  # ~0.5GB
            tensors.append(t)
            print(f"🔁 Allocation {i+1} → RAM saturée ? {is_ram_saturated()}")
    except RuntimeError as e:
        print("❌ Erreur d’allocation :", e)

    print("📊 RAM après allocation :", get_ram_status())

if __name__ == "__main__":
    simulate_memory_stress()
