# tests/test_scheduler_dynamic.py

import torch
import torch.nn as nn
from core.scheduler import SimpleScheduler

def dummy_block(name):
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = name
        def forward(self, x):
            return x + 1
    return Dummy()

def test_scheduler_routing():
    blocks = [dummy_block(f"Block{i}") for i in range(4)]
    scheduler = SimpleScheduler(blocks)
    input_tensor = torch.zeros(1, 10)
    output = scheduler.forward(input_tensor)
    assert output.shape == input_tensor.shape
    print("✅ Routage dynamique exécuté avec succès")

if __name__ == "__main__":
    test_scheduler_routing()
