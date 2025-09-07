# core/scheduler.py
"""
Simple scheduler qui exécute un modèle divisé en blocs
sur les GPU correspondants (CUDA / ROCm / MPS).
"""

import time
import torch
from .utils import assign_block_to_device


class SimpleScheduler:
    def __init__(self, blocks, schedule_interval=0.1, verbose=False):
        # Attribuer chaque bloc à son device hétérogène
        self.blocks = [assign_block_to_device(b, i) for i, b in enumerate(blocks)]
        self.schedule_interval = schedule_interval
        self.verbose = verbose

    def forward(self, input_ids):
        hidden = input_ids
        for idx, block in enumerate(self.blocks):
            if self.verbose:
                print(f"[Scheduler] Forward block {idx} sur {block.device}")
            hidden = block(hidden)
            # Si le bloc a un autre device, PyTorch déplace automatiquement
        return hidden

    def predict(self, input_ids):
        hidden = self.forward(input_ids)
        return torch.argmax(hidden, dim=-1)
