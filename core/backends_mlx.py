"""Adaptateur MLX : servir un modèle MLX (Apple Silicon) via le sous-processus `mlx_lm.server`.

MLX est le moteur natif d'Apple : il travaille directement dans la mémoire unifiée des
puces M1-M4 et lit les modèles au format MLX (dépôts `mlx-community/…` sur Hugging Face),
pas les GGUF. `mlx_lm.server` expose une API compatible OpenAI (/v1/completions,
/v1/chat/completions) : on le pilote comme llama-server, dans un sous-processus.

Choisi automatiquement sur Apple Silicon pour un modèle non-GGUF quand `mlx_lm` est
installé (`pip install mlx-lm`) ; `--backend mlx` le force. Les GGUF restent sur
llama.cpp (build Metal, téléchargé automatiquement).

Non mesuré sur un vrai Mac à ce jour : validé par la CI sur les runners macOS arm64 de
GitHub (voir .github/workflows/apple-silicon.yml).
"""
from __future__ import annotations

import atexit
import importlib.util
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import weakref
from typing import Any, Iterator, List, Optional

import requests as _requests

from core.backends import BaseLLMBackend
from core.llama_server_backend import (LlamaServerBackend, _die_with_parent, _free_port,
                                       _terminate_proc)

log = logging.getLogger("vramancer.backends.mlx")


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


class MlxServerBackend(LlamaServerBackend):
    """`mlx_lm.server` en sous-processus. Réutilise chat / chat_stream / generate de
    LlamaServerBackend : même API OpenAI de l'autre côté."""

    def __init__(self, model: str, port: Optional[int] = None, python: Optional[str] = None,
                 ready_timeout: int = 1800):
        # Pas d'appel à LlamaServerBackend.__init__ : rien à télécharger, pas de GGUF.
        self._model_path = model
        self._rpc_hosts: List[str] = []
        self._port = _free_port(port or int(os.environ.get("VRM_MLX_PORT", "8082")))
        self._base_url = f"http://127.0.0.1:{self._port}"
        cmd = [python or sys.executable, "-m", "mlx_lm.server", "--model", model,
               "--host", "127.0.0.1", "--port", str(self._port)]
        log.info("Démarrage de mlx_lm.server : %s", model)
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=self._stderr,
            preexec_fn=_die_with_parent if os.name == "posix" else None)
        # Premier lancement : téléchargement du modèle depuis Hugging Face → délai large.
        self._wait_ready(ready_timeout)
        self._finalizer = weakref.finalize(self, _terminate_proc, self._proc)
        atexit.register(self.shutdown)

    def _wait_ready(self, timeout: int = 1800):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if _requests.get(f"{self._base_url}/v1/models", timeout=2).status_code == 200:
                    log.info("mlx_lm.server prêt sur le port %d", self._port)
                    return
            except Exception:
                pass
            if self._proc and self._proc.poll() is not None:
                self._stderr.seek(0)
                err = self._stderr.read().decode("utf-8", "ignore")[-800:]
                raise RuntimeError(f"mlx_lm.server s'est arrêté : {err}")
            time.sleep(0.5)
        raise RuntimeError(f"mlx_lm.server pas prêt après {timeout} s")


class MlxServerAdapter(BaseLLMBackend):
    """`MlxServerBackend` exposé comme backend VRAMancer (backend TEXTE, comme llama-server)."""

    def __init__(self, model_name: str = None, cache_dir: str = None, **kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._server: Optional[MlxServerBackend] = None
        self.tokenizer = None

    def load_model(self, model_name: str = None, **kwargs) -> Any:
        if not mlx_available():
            raise RuntimeError("mlx_lm n'est pas installé : pip install mlx-lm")
        self.model_name = model_name or self.model_name
        self._server = MlxServerBackend(self.model_name)
        try:                                   # les dépôts MLX embarquent le tokenizer HF
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception:
            log.debug("tokenizer indisponible : usage compté approximativement", exc_info=True)
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

    def split_model(self, num_gpus: int, vram_per_gpu: Optional[List[int]] = None) -> List[Any]:
        """Mémoire unifiée : un seul « GPU », rien à découper côté VRAMancer."""
        return [self._server] if self._server else []

    def infer(self, inputs: Any) -> Any:
        raise NotImplementedError("MlxServerAdapter est un backend texte : utilise generate().")

    def _current(self) -> MlxServerBackend:
        if self._server is None:
            self.load_model(self.model_name)
        return self._server

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        return self._current().generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> Iterator[str]:
        return self._current().chat_stream(
            [{"role": "user", "content": prompt}], max_tokens=max_new_tokens, **kwargs)

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        return [self.generate(p, max_new_tokens=max_new_tokens, **kwargs) for p in prompts]


__all__ = ["MlxServerAdapter", "MlxServerBackend", "is_apple_silicon", "mlx_available"]
