"""Ollama backend integration for VRAMancer.

Extracted from core/backends.py for maintainability.
Communicates with a local Ollama server via its REST API.
Supports streaming via chunked JSON responses.
"""

import os
from typing import Any, List, Optional

from core.backends import BaseLLMBackend
from core.logger import LoggerAdapter


class OllamaBackend(BaseLLMBackend):
    """Ollama backend — communicates with a local Ollama server via REST API.
    """
    def __init__(self, real: bool = True):
        self.model = None
        self.model_name: Optional[str] = None
        self.log = LoggerAdapter("backend.ollama" + (".stub" if not real else ""))
        self.real = real
        self._base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def load_model(self, model_name: str, **kwargs):
        self.model_name = model_name
        if not self.real:
            self.model = {"name": model_name, "stub": True}
            return self.model
        try:
            import requests
            resp = requests.post(
                f"{self._base_url}/api/show",
                json={"name": model_name},
                timeout=10,
            )
            if resp.status_code == 200:
                self.model = model_name
                self.log.info(f"Ollama native async bridge ready for: {model_name}")
            else:
                self.log.warning(f"Ollama model '{model_name}' not found locally, will be pulled on first generate()")
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
        # Ollama manages physical GPUs natively via its Go scheduler
        return [self.model]

    def infer(self, inputs: Any):
        if self.model is None:
            raise RuntimeError("Modèle Ollama non chargé.")
        if not self.real:
            return {"text": "stub-ollama-output", "len_in": getattr(inputs, 'shape', '?')}
        prompt = inputs if isinstance(inputs, str) else str(inputs)
        return {"text": self.generate(prompt), "model": self.model}

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        """Generate text via Ollama REST API (Sync fallback)."""
        if self.model is None:
            raise RuntimeError("Modèle Ollama non chargé.")
        if not self.real:
            return f"[stub-ollama] {prompt[:50]}..."
        try:
            import requests
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": kwargs.get('temperature', 1.0),
                        "top_p": kwargs.get('top_p', 1.0),
                        "top_k": kwargs.get('top_k', 50),
                        "num_gpu": kwargs.get('num_gpu', -1),
                    },
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get('response', '')
            self.log.error(f"Ollama API error: {resp.status_code} {resp.text[:200]}")
            raise RuntimeError(f"Ollama API returned {resp.status_code}")
        except ImportError:
            raise RuntimeError("requests library required for Ollama backend")
        except Exception as e:
            self.log.error(f"Ollama generate failed: {e}")
            raise

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        """Stream tokens from Ollama via chunked JSON responses."""
        if not self.real:
            text = f"[stub-ollama] {prompt[:50]}..."
            for word in text.split(' '):
                yield word + ' '
            return
        try:
            import requests
            with requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": kwargs.get('temperature', 1.0),
                        "top_p": kwargs.get('top_p', 1.0),
                        "top_k": kwargs.get('top_k', 50),
                        "num_gpu": kwargs.get('num_gpu', -1),
                    },
                },
                timeout=120,
                stream=True,
            ) as resp:
                import json as _json
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = _json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except _json.JSONDecodeError:
                        continue
        except Exception as e:
            self.log.error(f"Ollama stream failed: {e}")
            raise
