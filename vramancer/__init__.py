"""VRAMancer — orchestration multi-GPU hétérogène (via accelerate/llama.cpp) + optims mesurées.

Drop-in (S1) :
    import vramancer; vramancer.patch()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B")
"""
from .dropin import patch, unpatch, is_patched, compute_max_memory

__all__ = ["patch", "unpatch", "is_patched", "compute_max_memory"]
