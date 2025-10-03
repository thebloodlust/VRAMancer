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
        # Simple placeholder GPU map (single GPU index 0)
        self._available_gpus = [0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if "on_start" in self.callbacks:
            self.callbacks["on_start"](0, x)

        for idx, block in enumerate(self.blocks):
            if "on_step" in self.callbacks:
                self.callbacks["on_step"](idx, x)

            meta = get_block_metadata(idx)
            out = self.router.route(block, x, index=idx, **meta)
            # Certains modèles HuggingFace retournent un objet avec attribut logits
            if hasattr(out, 'logits'):
                out = out.logits
            x = out

        if "on_end" in self.callbacks:
            self.callbacks["on_end"](len(self.blocks) - 1, x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Compatibility API attendu par MemoryBalancer / Orchestrator
    # ------------------------------------------------------------------
    def get_available_gpus(self):  # pragma: no cover - trivial
                """Retourne la liste des GPUs disponibles (placeholder struct).

                Structure attendue par MemoryBalancer:
                    [{"id": int, "total_vram_mb": int}, ...]
                """
                count = torch.cuda.device_count() if torch.cuda.is_available() else 1
                gpus = []
                real_sizes = None
                try:  # tentative via nvidia-ml-py
                    import pynvml  # type: ignore
                    pynvml.nvmlInit()
                    real_sizes = []
                    for i in range(count):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        real_sizes.append(int(mem.total/1024/1024))
                    pynvml.nvmlShutdown()
                except Exception:  # pragma: no cover - fallback path
                    real_sizes = None
                for idx in range(count):
                    total_mb = real_sizes[idx] if real_sizes and idx < len(real_sizes) else (16_000 if count == 1 else 24_000)
                    gpus.append({"id": idx, "total_vram_mb": total_mb})
                return gpus
