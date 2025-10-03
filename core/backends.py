# ------------------- Sélection dynamique du backend -------------------
def select_backend(backend_name: str = "auto"):
    """
    Sélectionne dynamiquement le backend LLM selon le nom ou l’environnement.
    - backend_name : "huggingface", "vllm", "ollama", ou "auto"
    """
    backend_name = (backend_name or "auto").lower()
    if backend_name == "huggingface":
        return HuggingFaceBackend()
    if backend_name == "vllm":
        try:
            import vllm  # noqa: F401
            return vLLMBackend()
        except ImportError:
            raise RuntimeError("vLLM n'est pas installé.")
    if backend_name == "ollama":
        try:
            import ollama  # noqa: F401
            return OllamaBackend()
        except ImportError:
            raise RuntimeError("Ollama n'est pas installé.")
    # Mode auto : priorité vLLM > Ollama > HuggingFace
    try:
        import vllm  # noqa: F401
        return vLLMBackend()
    except ImportError:
        pass
    try:
        import ollama  # noqa: F401
        return OllamaBackend()
    except ImportError:
        pass
    return HuggingFaceBackend()
# core/backends.py
"""
Abstraction unifiée pour les backends LLM (HuggingFace, vLLM, Ollama, etc.)
"""
from abc import ABC, abstractmethod
from typing import Any, List
from core.logger import LoggerAdapter

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
        for block in self.blocks:
            x = block(x)
        self.log.debug("Fin inférence")
        return x

# ------------------- vLLM Backend (squelette) -------------------
class vLLMBackend(BaseLLMBackend):
    def __init__(self):
        """Backend vLLM natif (nécessite le package vllm)."""
        self.model = None
        self.log = LoggerAdapter("backend.vllm")

    def load_model(self, model_name: str, **kwargs):
        """
        Charge un modèle via l’API vLLM (si installé).
        """
        try:
            from vllm import LLM
            self.model = LLM(model=model_name, **kwargs)
            return self.model
        except ImportError:
            self.log.error("vLLM n'est pas installé")
            raise RuntimeError("vLLM n’est pas installé. Faites pip install vllm.")
        except Exception as e:
            self.log.error(f"Erreur vLLM: {e}")
            raise RuntimeError(f"Erreur vLLM: {e}")

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        """
        vLLM gère le dispatch GPU en interne. Retourne un stub unique.
        """
        if self.model is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        return [self.model]

    def infer(self, inputs: Any):
        """
        Effectue une inférence via vLLM (API Python ou REST selon install).
        """
        if self.model is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        # Ex: self.model.generate(...)
        raise NotImplementedError("Intégration vLLM réelle à compléter selon votre install.")

# ------------------- Ollama Backend (squelette) -------------------
class OllamaBackend(BaseLLMBackend):
    def __init__(self):
        """Backend Ollama natif (nécessite Ollama en local ou via REST)."""
        self.model = None
        self.log = LoggerAdapter("backend.ollama")

    def load_model(self, model_name: str, **kwargs):
        """
        Charge un modèle Ollama (via REST ou SDK Python si dispo).
        """
        try:
            import requests
            # Ex: requests.post('http://localhost:11434/api/generate', ...)
            self.model = model_name
            return self.model
        except ImportError:
            self.log.error("requests n'est pas installé")
            raise RuntimeError("requests n’est pas installé. Faites pip install requests.")
        except Exception as e:
            self.log.error(f"Erreur Ollama: {e}")
            raise RuntimeError(f"Erreur Ollama: {e}")

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        """
        Ollama ne supporte pas le split GPU natif. Retourne un stub unique.
        """
        if self.model is None:
            raise RuntimeError("Modèle Ollama non chargé.")
        return [self.model]

    def infer(self, inputs: Any):
        """
        Effectue une inférence via Ollama (REST API ou SDK Python).
        """
        if self.model is None:
            raise RuntimeError("Modèle Ollama non chargé.")
        # Ex: requests.post(...)
        raise NotImplementedError("Intégration Ollama réelle à compléter selon votre install.")
