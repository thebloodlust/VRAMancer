"""Abstraction unifiée des backends LLM.

Ajouts :
- Fallback stub vLLM / Ollama si dépendances absentes (activer via env
  `VRM_BACKEND_ALLOW_STUB=1`).
- Hooks de tracking mémoire (injection hmem) déjà gérés côté HuggingFace.
"""
from abc import ABC, abstractmethod
from typing import Any, List
import os
from core.logger import LoggerAdapter
import hashlib

def select_backend(backend_name: str = "auto"):
    backend_name = (backend_name or "auto").lower()
    allow_stub = os.environ.get("VRM_BACKEND_ALLOW_STUB")
    if backend_name == "huggingface":
        return HuggingFaceBackend()
    if backend_name == "vllm":
        try:
            import vllm  # noqa: F401
            return vLLMBackend(real=True)
        except ImportError:
            if allow_stub: return vLLMBackend(real=False)
            raise RuntimeError("vLLM non installé (export VRM_BACKEND_ALLOW_STUB=1 pour stub)")
    if backend_name == "ollama":
        try:
            import ollama  # noqa: F401
            return OllamaBackend(real=True)
        except ImportError:
            if allow_stub: return OllamaBackend(real=False)
            raise RuntimeError("Ollama non installé (export VRM_BACKEND_ALLOW_STUB=1 pour stub)")
    # auto
    try:
        import vllm  # noqa: F401
        return vLLMBackend(real=True)
    except ImportError:
        if allow_stub:
            return vLLMBackend(real=False)
    try:
        import ollama  # noqa: F401
        return OllamaBackend(real=True)
    except ImportError:
        if allow_stub:
            return OllamaBackend(real=False)
    return HuggingFaceBackend()

class BaseLLMBackend(ABC):
    """
    Interface générique pour tous les backends LLM.
    """
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> Any:
        pass

    @abstractmethod
    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None) -> List[Any]:
        """
        Découpe le modèle en blocs selon le nombre de GPUs et la VRAM disponible.
        """
        pass

    @abstractmethod
    def infer(self, inputs: Any) -> Any:
        pass

# ------------------- HuggingFace Backend -------------------
class HuggingFaceBackend(BaseLLMBackend):
    def __init__(self):
        self.model = None
        self.blocks = None
        self.log = LoggerAdapter("backend.hf")
        self.hmem = None  # référence injectée optionnelle

    def load_model(self, model_name: str, **kwargs):
        from transformers import AutoModelForCausalLM
        self.log.info(f"Chargement modèle HuggingFace: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        return self.model

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        # TODO: utiliser split_model_into_blocks adapté à la VRAM réelle
        from core.model_splitter import split_model_into_blocks
        self.log.debug(f"Découpage en blocs sur {num_gpus} GPUs")
        self.blocks = split_model_into_blocks(self.model, num_gpus, vram_per_gpu)
        return self.blocks

    def infer(self, inputs: Any):
        # Simple forward sur tous les blocs (à adapter pour pipeline multi-GPU)
        x = inputs
        self.log.debug("Début inférence séquentielle sur blocs")
        if not self.blocks:
            raise RuntimeError("Blocs non initialisés")
    from core.memory_block import MemoryBlock
        for block in self.blocks:
            x = block(x)
            # Hook accès mémoire : chaque passage = touch + éventuelle promotion
            if self.hmem:
                mb = MemoryBlock(size_mb=getattr(block, 'size_mb', 128), gpu_id=0, status="allocated")
                # On reconstruit un id stable via id(block) hashé
                bid = hashlib.sha1(str(id(block)).encode()).hexdigest()
                # Enregistrer si pas présent
                if bid not in self.hmem.registry:
                    mb.id = bid
                    self.hmem.register_block(mb, "L1")
                else:
                    mb.id = bid
                self.hmem.touch(mb)
                self.hmem.promote_policy(mb)
        self.log.debug("Fin inférence")
        return x

# ------------------- vLLM Backend (squelette) -------------------
class vLLMBackend(BaseLLMBackend):
    def __init__(self, real: bool = True):
        self.model = None
        self.log = LoggerAdapter("backend.vllm" + (".stub" if not real else ""))
        self.real = real

    def load_model(self, model_name: str, **kwargs):
        if not self.real:
            self.model = {"name": model_name, "stub": True}
            return self.model
        try:
            from vllm import LLM
            self.model = LLM(model=model_name, **kwargs)
            return self.model
        except Exception as e:
            if os.environ.get("VRM_BACKEND_ALLOW_STUB"):
                self.log.warning(f"Fallback stub vLLM: {e}")
                self.real = False
                self.model = {"name": model_name, "stub": True}
                return self.model
            raise

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        if self.model is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        return [self.model]

    def infer(self, inputs: Any):
        if self.model is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        if not self.real:
            try:
                import torch
                return torch.zeros_like(inputs)
            except Exception:
                return inputs
        raise NotImplementedError("Intégration vLLM réelle manquante (installer vllm).")

# ------------------- Ollama Backend (squelette) -------------------
class OllamaBackend(BaseLLMBackend):
    def __init__(self, real: bool = True):
        self.model = None
        self.log = LoggerAdapter("backend.ollama" + (".stub" if not real else ""))
        self.real = real

    def load_model(self, model_name: str, **kwargs):
        if not self.real:
            self.model = {"name": model_name, "stub": True}
            return self.model
        try:
            import requests  # noqa: F401
            self.model = model_name
            return self.model
        except Exception as e:
            if os.environ.get("VRM_BACKEND_ALLOW_STUB"):
                self.log.warning(f"Fallback stub Ollama: {e}")
                self.real = False
                self.model = {"name": model_name, "stub": True}
                return self.model
            raise

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        if self.model is None:
            raise RuntimeError("Modèle Ollama non chargé.")
        return [self.model]

    def infer(self, inputs: Any):
        if self.model is None:
            raise RuntimeError("Modèle Ollama non chargé.")
        if not self.real:
            return {"text": "stub-ollama-output", "len_in": getattr(inputs, 'shape', '?')}
        raise NotImplementedError("Intégration Ollama réelle manquante (installer Ollama).")
