# core/scheduler.py
"""
Dynamic scheduler that routes input through a chain of modules,
each executed on the optimal device (GPU, CPU, RAM, NVMe, or network)
based on its importance and estimated size.
"""

import torch
from typing import Iterable, Callable, Any
from core.block_router import BlockRouter
from core.block_metadata import get_block_metadata

class SimpleScheduler:
    """
    Forward a tensor through a chain of modules with dynamic routing.

    Each block is executed on the best available device depending on:
    - GPU availability
    - RAM pressure
    - Block size and importance
    - Fallback to NVMe or remote execution

    Parameters
    ----------
    blocks : Iterable[torch.nn.Module]
        The chain of modules.
    callbacks : dict[str, Callable[[int, torch.Tensor], None]] | None
        Optional callbacks (e.g. ``on_start``, ``on_end``) that receive the
        current block index and the intermediate tensor.
    """

    def __init__(
        self,
        blocks: Iterable[torch.nn.Module],
        callbacks: dict[str, Callable[[int, torch.Tensor], None]] | None = None,
    ) -> None:
        self.blocks = list(blocks)
        self.callbacks = callbacks or {}
        self.router = BlockRouter()

    # ----------------------------------------------------------------------
    # 1️⃣  Forward pass
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run *x* through every block sequentially with dynamic routing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[batch, seq_len]` (or whatever the model expects).

        Returns
        -------
        torch.Tensor
            Output of the last block.
        """
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

    # ----------------------------------------------------------------------
    # 2️⃣  Convenience predict (for causal LM)
    # ----------------------------------------------------------------------
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Same as :meth:`forward` but returns the class indices (argmax)
        – useful for inference on causal language models.
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)
