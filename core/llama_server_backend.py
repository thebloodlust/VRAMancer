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

# GitHub release assets. Vérifié le 2026-09-22 sur la release b11112 : upstream
# publie des .tar.gz pour Linux/macOS (plus des .zip), le nom porte l'accélérateur
# (cuda-12.8, vulkan, rocm…), et la release « latest » est un tag de version
# (v0.4.1) qui ne contient QU'UN nightly-tag.txt — il faut donc résoudre le vrai
# tag de build bNNNNN avant de construire une URL.
_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
_RELEASE_DL = "https://github.com/ggml-org/llama.cpp/releases/download"
_ASSET_MAP = {
    "linux-cuda":   "llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "linux-vulkan": "llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz",
    "linux-cpu":    "llama-{tag}-bin-ubuntu-x64.tar.gz",
    "darwin-arm":   "llama-{tag}-bin-macos-arm64.tar.gz",
    "darwin-x86":   "llama-{tag}-bin-macos-x64.tar.gz",
    "windows":      "llama-{tag}-bin-win-cuda-12.4-x64.zip",
}


def _has_amd_gpu() -> bool:
    """Carte AMD présente ? (sysfs amdgpu, sans dépendre de ROCm)."""
    try:
        from core.amd_sysfs import has_amd_gpu
        return has_amd_gpu()
    except Exception:
        return False


def _platform_key() -> str:
    """Quel binaire llama.cpp pour cette machine.

    Sur Linux, une carte AMD sans NVIDIA prenait le build CPU : mesuré le
    2026-09-22 sur RX 7900 XT + Qwen3.6-35B-A3B Q4_K_M, cela coûte 6.59 tok/s
    (CPU) contre 37.78 tok/s (Vulkan), soit 5.7×. D'où la détection AMD → Vulkan.
    """
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "darwin-arm" if platform.machine() == "arm64" else "darwin-x86"
    if sys_name == "windows":
        return "windows"
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return "linux-cuda"
    except Exception:
        pass
    if _has_amd_gpu():
        return "linux-vulkan"
    return "linux-cpu"


def _latest_build_tag() -> str:
    """Dernier tag de BUILD (bNNNNN), pas le tag de version."""
    import re
    try:
        resp = _requests.get(f"{_RELEASES_API}?per_page=10", timeout=15)
        for rel in resp.json():
            tag = rel.get("tag_name", "")
            if re.fullmatch(r"b\d+", tag):
                return tag
    except Exception as e:
        log.warning("Impossible de résoudre le tag llama.cpp (%s)", e)
    raise RuntimeError(
        "Aucun tag de build llama.cpp trouvé. Télécharge un binaire manuellement "
        "depuis https://github.com/ggml-org/llama.cpp/releases et pointe "
        "VRM_LLAMA_SERVER_BIN dessus."
    )


def get_or_download_binary() -> Path:
    """Chemin du binaire llama-server (téléchargé si absent).

    `VRM_LLAMA_SERVER_BIN` court-circuite tout : utile pour pointer un build
    local (ex. un build Vulkan compilé soi-même) sans rien télécharger.
    """
    override = os.environ.get("VRM_LLAMA_SERVER_BIN")
    if override:
        p = Path(override).expanduser()
        if p.is_dir():
            p = p / "llama-server"
        if not p.exists():
            raise RuntimeError(f"VRM_LLAMA_SERVER_BIN pointe sur un binaire inexistant: {p}")
        log.info("llama-server (VRM_LLAMA_SERVER_BIN): %s", p)
        return p

    BINARY_DIR.mkdir(parents=True, exist_ok=True)

    existing = _find_server_binary()
    if existing:
        log.info("llama-server binary found: %s", existing)
        return existing

    log.info("Downloading llama-server binary…")
    _download_release_binary()

    ready = _find_server_binary()
    if ready:
        ready.chmod(0o755)
        log.info("llama-server ready: %s", ready)
        return ready

    raise RuntimeError(
        f"llama-server binary not found after download in {BINARY_DIR}. "
        "Download manually from https://github.com/ggml-org/llama.cpp/releases "
        f"and place in {BINARY_DIR}"
    )


def _download_release_binary():
    """Télécharge la release llama.cpp et extrait l'archive TELLE QUELLE.

    Deux pièges vérifiés le 2026-09-22 sur b11112 :
      - les builds sont dynamiques (libggml-*.so, libllama-*.so à côté du binaire) :
        extraire seulement `llama-server` donne un exécutable qui ne démarre pas ;
      - les bibliothèques sont versionnées AVEC des liens symboliques
        (libllama-common.so.0 -> .so.0.4.1) : les aplatir en fichiers casse
        l'édition de liens. On préserve donc l'arborescence et les liens.
    """
    import io
    import tarfile
    import zipfile

    BINARY_DIR.mkdir(parents=True, exist_ok=True)
    tag = _latest_build_tag()
    pk = _platform_key()
    asset = _ASSET_MAP.get(pk, _ASSET_MAP["linux-cpu"]).format(tag=tag)
    url = f"{_RELEASE_DL}/{tag}/{asset}"

    log.info("Downloading %s (%s) …", url, pk)
    resp = _requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    data = resp.content

    dest_root = BINARY_DIR / f"{tag}-{pk}"
    if dest_root.exists():
        import shutil
        shutil.rmtree(dest_root, ignore_errors=True)
    dest_root.mkdir(parents=True, exist_ok=True)

    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(dest_root)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            tf.extractall(dest_root, filter="data")

    found = _find_server_binary()
    if found:
        found.chmod(0o755)
        for sibling in found.parent.glob("llama-*"):
            try:
                sibling.chmod(0o755)
            except Exception:
                pass
    log.info("llama.cpp %s (%s) extrait dans %s", tag, pk, dest_root)


def _find_server_binary():
    """Cherche llama-server dans BINARY_DIR (récursif : les archives ont un bin/)."""
    for name in ("llama-server", "llama-server.exe", "server", "server.exe"):
        direct = BINARY_DIR / name
        if direct.is_file():
            return direct
    for cand in sorted(BINARY_DIR.rglob("llama-server*")):
        if cand.is_file():
            return cand
    return None


def _runtime_env(binary) -> dict:
    """Env d'exécution : les libggml-*.so vivent à côté du binaire, pas dans /usr/lib."""
    env = dict(os.environ)
    libdir = str(Path(binary).parent)
    env["LD_LIBRARY_PATH"] = (libdir + os.pathsep + env["LD_LIBRARY_PATH"]
                              if env.get("LD_LIBRARY_PATH") else libdir)
    return env


def _server_help(binary) -> str:
    """`--help` du binaire, pour s'adapter aux options qui ont changé de forme."""
    try:
        r = subprocess.run([str(binary), "--help"], capture_output=True,
                           timeout=30, env=_runtime_env(binary))
        return (r.stdout + r.stderr).decode("utf-8", "ignore")
    except Exception as e:
        log.debug("llama-server --help indisponible: %s", e)
        return ""


def _compat_flags(binary) -> List[str]:
    """Options dont la FORME a changé selon la version de llama.cpp.

    Vérifié le 2026-09-22 sur b11112 : `--flash-attn` exige désormais une valeur
    (le passer nu avale l'argument suivant et le serveur refuse de démarrer), et
    `--no-mmap` a été remplacé par `--load-mode none`. On lit `--help` plutôt que
    de supposer, pour rester compatible avec un binaire local plus ancien
    (VRM_LLAMA_SERVER_BIN).
    """
    h = _server_help(binary)
    flags: List[str] = []
    if "--flash-attn" in h:
        flags += ["--flash-attn", "on"] if "[on|off|auto]" in h else ["--flash-attn"]
    if "--load-mode" in h:
        flags += ["--load-mode", "none"]
    elif "--no-mmap" in h or not h:
        flags += ["--no-mmap"]
    if "--log-disable" in h:
        flags += ["--log-disable"]
    return flags


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
            # -1 = toutes les couches. Attention : si le modèle dépasse la VRAM,
            # le pilote déborde en mémoire hôte et le débit s'effondre (mesuré sur
            # RX 7900 XT : 37.78 -> 8.95 tok/s). VRM_LLAMA_NGL permet de plafonner.
            "--n-gpu-layers", os.environ.get("VRM_LLAMA_NGL", "-1"),
        ]
        cmd += _compat_flags(binary)

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
        env = _runtime_env(binary)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
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
