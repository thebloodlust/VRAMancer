# core/scheduler.py
"""
Simple scheduler that forwards a batch of input_ids through
a list of modules, each potentially on a different device.
"""

import torch
from typing import Iterable, List, Callable, Any
from .utils import assign_block_to_device


class SimpleScheduler:
    """
    Forward a tensor through a chain of modules.

    Each module is automatically moved to the GPU that matches
    its logical index (0 → cuda, 1 → rocm, 2 → mps, …).

    Parameters
    ----------
    blocks : Iterable[torch.nn.Module]
        The chain of modules.
    callbacks : dict[str, Callable[[Any], None]] | None
        Optional callbacks (e.g. ``on_start``, ``on_end``) that receive the
        current block index and the intermediate tensor.
    """

    def __init__(
        self,
        blocks: Iterable[torch.nn.Module],
        callbacks: dict[str, Callable[[int, torch.Tensor], None]] | None = None,
    ) -> None:
        self.blocks = [assign_block_to_device(b, idx) for idx, b in enumerate(blocks)]
        self.callbacks = callbacks or {}

    # ----------------------------------------------------------------------
    # 1️⃣  Forward pass
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run *x* through every block sequentially.

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
            x = block(x)
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
