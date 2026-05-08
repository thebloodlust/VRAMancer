"""LlamaServerBackend — wraps llama-server binary with optional RPC nodes.

Downloads the llama.cpp release binary on first use.
Supports --rpc for remote GPU nodes (MacBook M4, RTX 4060 laptop, etc.)

Architecture:
    VRAMancer server.py
        └─ LlamaServerBackend
            ├─ spawns: llama-server --model *.gguf --rpc [mac]:50052 ...
            └─ proxies: HTTP requests → localhost:8081 (OpenAI-compat API)

Remote node setup:
    MacBook (brew):  brew install llama.cpp && llama-rpc-server --host 0.0.0.0 --port 50052
    Laptop (pip):    python -m llama_cpp.server.rpc --host 0.0.0.0 --port 50052
    Or binary:       llama-rpc-server --host 0.0.0.0 --port 50052
"""
import gc
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Iterator, List, Optional

import requests as _requests

log = logging.getLogger("vramancer.llama_server")

# ── Constants ──────────────────────────────────────────────────────────────────

BINARY_DIR  = Path.home() / ".cache" / "vramancer" / "bin"
SERVER_PORT = int(os.environ.get("VRM_LLAMA_SERVER_PORT", "8081"))

# GitHub release asset names per platform
_RELEASE_BASE = "https://github.com/ggml-org/llama.cpp/releases/latest/download"
_ASSET_MAP = {
    "linux-cuda":  "llama-{tag}-bin-ubuntu-x64.zip",
    "linux-cpu":   "llama-{tag}-bin-ubuntu-x64.zip",
    "darwin-arm":  "llama-{tag}-bin-macos-arm64.zip",
    "darwin-x86":  "llama-{tag}-bin-macos-x64.zip",
    "windows":     "llama-{tag}-bin-win-cuda-cu12.2.0-x64.zip",
}


def _platform_key() -> str:
    sys = platform.system().lower()
    if sys == "darwin":
        return "darwin-arm" if platform.machine() == "arm64" else "darwin-x86"
    if sys == "windows":
        return "windows"
    # Linux: check CUDA
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return "linux-cuda"
    except Exception:
        return "linux-cpu"


def get_or_download_binary() -> Path:
    """Return path to llama-server binary, downloading if needed."""
    BINARY_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already present
    for name in ("llama-server", "llama-server.exe", "server", "server.exe"):
        p = BINARY_DIR / name
        if p.exists():
            log.info("llama-server binary found: %s", p)
            return p

    log.info("Downloading llama-server binary…")
    _download_release_binary()

    # Re-check
    for name in ("llama-server", "llama-server.exe", "server"):
        p = BINARY_DIR / name
        if p.exists():
            p.chmod(0o755)
            log.info("llama-server ready: %s", p)
            return p

    raise RuntimeError(
        f"llama-server binary not found after download in {BINARY_DIR}. "
        "Download manually from https://github.com/ggml-org/llama.cpp/releases "
        f"and place in {BINARY_DIR}"
    )


def _download_release_binary():
    """Download latest llama.cpp release and extract llama-server."""
    import urllib.request
    import zipfile
    import io

    # Get latest tag
    try:
        resp = _requests.get(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            timeout=10,
        )
        tag = resp.json()["tag_name"]
    except Exception:
        tag = "b5000"  # fallback

    pk = _platform_key()
    asset = _ASSET_MAP.get(pk, _ASSET_MAP["linux-cpu"]).format(tag=tag)
    url = f"{_RELEASE_BASE}/{asset}"

    log.info("Downloading %s …", url)
    try:
        resp = _requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        data = resp.content
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if "llama-server" in member or member.endswith("/server"):
                    fname = Path(member).name
                    dest = BINARY_DIR / fname
                    dest.write_bytes(zf.read(member))
                    dest.chmod(0o755)
                    log.info("Extracted: %s", dest)
    except Exception as e:
        log.error("Binary download failed: %s — install manually", e)
        raise


# ── Backend class ─────────────────────────────────────────────────────────────

class LlamaServerBackend:
    """Manages a llama-server subprocess with optional RPC remote nodes.

    Usage:
        backend = LlamaServerBackend.from_hub(
            repo_id="unsloth/Qwen3-Coder-Next-GGUF",
            filename="Qwen3-Coder-Next-UD-Q4_K_XL.gguf",
            rpc_hosts=["192.168.1.29:50052"],   # MacBook M4
            num_local_gpus=2,
        )
    """

    def __init__(
        self,
        model_path: str,
        rpc_hosts: Optional[List[str]] = None,
        num_local_gpus: int = 2,
        n_ctx: int = 16384,
        server_port: int = SERVER_PORT,
        binary_path: Optional[str] = None,
    ):
        self._model_path  = model_path
        self._rpc_hosts   = rpc_hosts or []
        self._port        = server_port
        self._proc: Optional[subprocess.Popen] = None
        self._base_url    = f"http://127.0.0.1:{server_port}"

        binary = Path(binary_path) if binary_path else get_or_download_binary()

        cmd = [
            str(binary),
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(server_port),
            "--ctx-size", str(n_ctx),
            "--n-gpu-layers", "-1",
            "--flash-attn",
            "--no-mmap",
            "--log-disable",
        ]

        # Tensor split across local GPUs proportional to VRAM
        if num_local_gpus > 1:
            split = _local_tensor_split(num_local_gpus)
            if split:
                cmd += ["--tensor-split", ",".join(str(s) for s in split)]

        # RPC remote nodes
        if self._rpc_hosts:
            cmd += ["--rpc", ",".join(self._rpc_hosts)]
            log.info("RPC nodes: %s", self._rpc_hosts)

        log.info("Starting llama-server: %s", " ".join(cmd[:6]) + " …")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._wait_ready()

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        filename: str,
        rpc_hosts: Optional[List[str]] = None,
        num_local_gpus: int = 2,
        n_ctx: int = 16384,
    ) -> "LlamaServerBackend":
        from core.llama_backend import download_gguf, _find_cached, KNOWN_FILES
        fn = filename or KNOWN_FILES.get(repo_id)
        if not fn:
            raise ValueError(f"No filename for {repo_id}")
        local = _find_cached(repo_id, fn) or download_gguf(repo_id, fn)
        return cls(local, rpc_hosts=rpc_hosts, num_local_gpus=num_local_gpus, n_ctx=n_ctx)

    # ── Readiness ────────────────────────────────────────────────────────────

    def _wait_ready(self, timeout: int = 120):
        """Poll /health until llama-server is ready."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = _requests.get(f"{self._base_url}/health", timeout=2)
                if r.status_code == 200:
                    log.info("llama-server ready on port %d", self._port)
                    return
            except Exception:
                log.debug("llama-server health check failed", exc_info=True)
            if self._proc and self._proc.poll() is not None:
                err = self._proc.stderr.read().decode()[:500]
                raise RuntimeError(f"llama-server crashed: {err}")
            time.sleep(0.5)
        raise RuntimeError(f"llama-server not ready after {timeout}s")

    # ── Inference ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> str:
        r = _requests.post(
            f"{self._base_url}/v1/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens,
                  "temperature": temperature, "top_p": top_p,
                  "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        messages: List[dict],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> Iterator[str]:
        with _requests.post(
            f"{self._base_url}/v1/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens,
                  "temperature": temperature, "top_p": top_p,
                  "stream": True},
            stream=True,
            timeout=300,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode()
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    text = chunk["choices"][0]["delta"].get("content", "")
                    if text:
                        yield text
                except Exception:
                    log.debug("Stream chunk parse failed", exc_info=True)

    def generate(self, prompt: str, max_new_tokens: int = 512, **kw) -> str:
        r = _requests.post(
            f"{self._base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_new_tokens, **kw},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        gc.collect()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _local_tensor_split(num_gpus: int) -> Optional[List[float]]:
    try:
        import torch
        return [
            round(torch.cuda.mem_get_info(i)[1] / 1e9, 1)
            for i in range(min(num_gpus, torch.cuda.device_count()))
        ]
    except Exception:
        return None
