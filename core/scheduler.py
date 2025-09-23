# core/scheduler.py

import torch
from typing import Iterable, Callable, Any
from core.block_router import BlockRouter
from core.block_metadata import get_block_metadata

class SimpleScheduler:
    def __init__(
        self,
        blocks: Iterable[torch.nn.Module],
        callbacks: dict[str, Callable[[int, torch.Tensor], None]] | None = None,
    ) -> None:
        self.blocks = list(blocks)
        self.callbacks = callbacks or {}
        self.router = BlockRouter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if "on_start" in self.callbacks:
            self.callbacks["on_start"](0, x)

        for idx, block in enumerate(self.blocks):
            if "on_step" in self.callbacks:
                self.callbacks["on_step"](idx, x)

            meta = get_block_metadata(idx)
            x = self.router.route(block, x, index=idx, **meta)

        if "on_end" in self.callbacks:
            self.callbacks["on_end"](len(self.blocks) - 1, x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)
