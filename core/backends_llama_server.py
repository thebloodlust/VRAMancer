"""Adaptateur : servir un GGUF via le SOUS-PROCESSUS llama-server.

Pourquoi : le paquet `llama-cpp-python` installé dans l'environnement est compilé
pour un accélérateur donné (ici CUDA). Sur une machine AMD, il s'importe très bien
mais `llama_supports_gpu_offload()` renvoie False : le modèle est alors chargé
**entièrement en CPU**, sans un mot, pendant que la carte reste à 1 % d'occupation.

Mesuré le 2026-09-22 sur RX 7900 XT + Qwen3.6-35B-A3B Q4_K_M :
  - `vramancer serve` via le binding in-process : **2.0 tok/s**, VRAM utilisée 335 MiB ;
  - le même modèle via llama-server Vulkan : **37.8 tok/s**, VRAM 19.8 GiB.

Cet adaptateur donne à `LlamaServerBackend` (qui télécharge le bon build : Vulkan
pour AMD, CUDA pour NVIDIA) l'interface attendue par le pipeline, pour que le choix
de backend puisse basculer dessus automatiquement.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterator, List, Optional

from core.backends import BaseLLMBackend

logger = logging.getLogger("vramancer.backends.llama_server")


class _ServerTokenizer:
    """Tokenizer adossé à l'endpoint `/tokenize` de llama-server.

    Sans lui, l'API retombe sur `len(text.split())` pour remplir `usage` : un
    compte de MOTS, qui sous-estime le code d'un facteur ~3 (mesuré le
    2026-09-23 : 2 851 annoncés pour 8 454 tokens réels). Un agent qui pilote son
    budget de contexte avec `usage` croirait avoir de la marge qu'il n'a pas.
    """

    def __init__(self, base_url: str):
        self._url = base_url.rstrip("/") + "/tokenize"

    def encode(self, text: str, **_kw) -> List[int]:
        import requests
        r = requests.post(self._url, json={"content": text}, timeout=30)
        r.raise_for_status()
        return r.json()["tokens"]

    def __call__(self, text: str, **kw):
        return {"input_ids": self.encode(text, **kw)}


class LlamaServerAdapter(BaseLLMBackend):
    """`LlamaServerBackend` exposé comme un backend VRAMancer standard.

    Limites assumées : pas de `split_model` réel (c'est llama-server qui répartit
    via --tensor-split) et `infer()` n'est pas fourni — ce backend est un backend
    de TEXTE (prompt → texte), comme vLLM/Ollama, pas un backend de tenseurs.
    """

    def __init__(self, model_name: str = None, cache_dir: str = None, **kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._server = None
        self.tokenizer = None          # renseigné par load_model() (/tokenize)
        self._num_gpus = int(kwargs.get("num_gpus", 1))

    # ── Cycle de vie ─────────────────────────────────────────────────────────
    def load_model(self, model_name: str = None, **kwargs) -> Any:
        from core.llama_server_backend import LlamaServerBackend
        path = model_name or self.model_name
        n_ctx = int(os.environ.get("VRM_N_CTX", kwargs.get("n_ctx", 8192)))
        num_gpus = int(kwargs.get("num_gpus", self._num_gpus))
        logger.info("llama-server (sous-processus) : %s, n_ctx=%d, gpus=%d",
                    path, n_ctx, num_gpus)
        # Nœuds distants : VRM_RPC_HOSTS explicite, sinon ceux qui ont rejoint via
        # `vramancer invite` et répondent maintenant (VRM_JOINED_NODES=0 pour ignorer).
        rpc = [h.strip() for h in os.environ.get("VRM_RPC_HOSTS", "").split(",") if h.strip()]
        if not rpc and os.environ.get("VRM_JOINED_NODES", "1") != "0":
            try:
                from core.join import joined_rpc_hosts
                rpc = joined_rpc_hosts()
            except Exception:
                logger.debug("registre des nœuds illisible", exc_info=True)
        if rpc:
            logger.info("Nœuds RPC : %s", ", ".join(rpc))
        self._server = LlamaServerBackend(
            path, rpc_hosts=rpc or None, num_local_gpus=num_gpus, n_ctx=n_ctx,
            server_port=int(os.environ.get("VRM_LLAMA_SERVER_PORT", "8081")),
        )
        self.model_name = path
        self.tokenizer = _ServerTokenizer(self._server._base_url)
        return self._server

    def shutdown(self):
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ── Interface backend ────────────────────────────────────────────────────
    def split_model(self, num_gpus: int, vram_per_gpu: Optional[List[int]] = None) -> List[Any]:
        """Pas de découpe côté VRAMancer : llama-server le fait via --tensor-split."""
        self._num_gpus = num_gpus
        return [self._server] if self._server else []

    def infer(self, inputs: Any) -> Any:
        raise NotImplementedError(
            "LlamaServerAdapter est un backend texte (prompt -> texte) : "
            "utilise generate(). Pour de l'inférence tenseur, prends le backend "
            "HuggingFace."
        )

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        if self._server is None:
            self.load_model(self.model_name)
        return self._server.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> Iterator[str]:
        if self._server is None:
            self.load_model(self.model_name)
        return self._server.chat_stream(
            [{"role": "user", "content": prompt}], max_tokens=max_new_tokens, **kwargs)

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        """Séquentiel : llama-server gère ses propres slots, on ne réordonne rien ici."""
        return [self.generate(p, max_new_tokens=max_new_tokens, **kwargs) for p in prompts]


__all__ = ["LlamaServerAdapter"]
