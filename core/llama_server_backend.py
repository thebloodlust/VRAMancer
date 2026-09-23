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
import atexit
import gc
import json
import logging
import socket
import weakref
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator, List, Optional

import requests as _requests

log = logging.getLogger("vramancer.llama_server")

# ── Constants ──────────────────────────────────────────────────────────────────

BINARY_DIR  = Path.home() / ".cache" / "vramancer" / "bin"
# Fork PrismML (modèles ternaires / 1 bit « Bonsai ») : rangé À PART, sinon la recherche
# récursive de BINARY_DIR pourrait le choisir pour un modèle ordinaire.
PRISM_BINARY_DIR = Path.home() / ".cache" / "vramancer" / "bin-prism"
SERVER_PORT = int(os.environ.get("VRM_LLAMA_SERVER_PORT", "8081"))

# GitHub release assets. Vérifié le 2026-09-22 sur la release b11112 : upstream
# publie des .tar.gz pour Linux/macOS (plus des .zip), le nom porte l'accélérateur
# (cuda-12.8, vulkan, rocm…), et la release « latest » est un tag de version
# (v0.4.1) qui ne contient QU'UN nightly-tag.txt — il faut donc résoudre le vrai
# tag de build bNNNNN avant de construire une URL.
_ASSET_MAP = {
    "linux-cuda":   "llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "linux-vulkan": "llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz",
    "linux-cpu":    "llama-{tag}-bin-ubuntu-x64.tar.gz",
    "darwin-arm":   "llama-{tag}-bin-macos-arm64.tar.gz",
    "darwin-x86":   "llama-{tag}-bin-macos-x64.tar.gz",
    "windows":      "llama-{tag}-bin-win-cuda-12.4-x64.zip",
}
# Même convention de noms chez PrismML, sauf le build CUDA Linux (vérifié sur
# prism-b10709-9a9394a, 2026-09-23). Certaines de leurs releases n'ont que les
# cudart Windows : on prend la plus récente qui contient VRAIMENT l'asset voulu.
_PRISM_ASSET_MAP = dict(_ASSET_MAP, **{
    "linux-cuda": "llama-{tag}-bin-linux-cuda-12.8-x64.tar.gz",
    "windows":    "llama-{tag}-bin-win-vulkan-x64.zip",
    "linux-rocm": "llama-{tag}-bin-ubuntu-rocm-7.2-x64.tar.gz",
})
_FLAVORS = {
    "upstream": {"repo": "ggml-org/llama.cpp", "tag_re": r"b\d+",
                 "dir": BINARY_DIR, "assets": _ASSET_MAP},
    "prism":    {"repo": "PrismML-Eng/llama.cpp", "tag_re": r"prism-b\d+-[0-9a-f]+",
                 "dir": PRISM_BINARY_DIR, "assets": _PRISM_ASSET_MAP},
}
# Types de tenseurs privés du fork PrismML (ggml.h du fork : PQ2_0, PTQ1_0). Le
# llama.cpp officiel refuse de charger ces fichiers (vérifié b11112).
PRISM_TENSOR_TYPES = {142, 143}


def _has_amd_gpu() -> bool:
    """Carte AMD présente ? (sysfs amdgpu, sans dépendre de ROCm)."""
    try:
        from core.amd_sysfs import has_amd_gpu
        return has_amd_gpu()
    except Exception:
        return False


def _has_nvidia() -> bool:
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
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
    nvidia = _has_nvidia()
    amd = _has_amd_gpu()
    if nvidia and not amd:
        return "linux-cuda"
    if amd:
        # NVIDIA + AMD : seul Vulkan voit les deux cartes (le build CUDA ignore l'AMD).
        return "linux-vulkan"
    return "linux-cpu"


def _latest_build_tag(flavor: str = "upstream", asset_key: Optional[str] = None) -> str:
    """Dernier tag de BUILD (bNNNNN), pas le tag de version.

    Avec `asset_key`, seulement une release qui publie cet asset (les releases
    PrismML sont parfois incomplètes).
    """
    import re
    fl = _FLAVORS[flavor]
    try:
        resp = _requests.get(f"https://api.github.com/repos/{fl['repo']}/releases?per_page=10",
                             timeout=15)
        for rel in resp.json():
            tag = rel.get("tag_name", "")
            if not re.fullmatch(fl["tag_re"], tag):
                continue
            if asset_key:
                want = fl["assets"][asset_key].format(tag=tag)
                if want not in {a.get("name") for a in rel.get("assets", [])}:
                    continue
            return tag
    except Exception as e:
        log.warning("Impossible de résoudre le tag llama.cpp (%s)", e)
    raise RuntimeError(
        "Aucun tag de build llama.cpp trouvé. Télécharge un binaire manuellement "
        "depuis https://github.com/ggml-org/llama.cpp/releases et pointe "
        "VRM_LLAMA_SERVER_BIN dessus."
    )


def get_or_download_binary(model_path=None) -> Path:
    """Chemin du binaire llama-server (téléchargé si absent).

    `VRM_LLAMA_SERVER_BIN` court-circuite tout : utile pour pointer un build
    local (ex. un build Vulkan compilé soi-même) sans rien télécharger.
    Un modèle PrismML (tenseurs ternaires) prend le fork PrismML
    (`VRM_PRISM_SERVER_BIN` pour un build local).
    """
    flavor = "prism" if model_path and needs_prism_fork(model_path) else "upstream"
    override = os.environ.get("VRM_PRISM_SERVER_BIN" if flavor == "prism"
                              else "VRM_LLAMA_SERVER_BIN")
    if override:
        p = Path(override).expanduser()
        if p.is_dir():
            p = p / "llama-server"
        if not p.exists():
            raise RuntimeError(f"{'VRM_PRISM_SERVER_BIN' if flavor == 'prism' else 'VRM_LLAMA_SERVER_BIN'}"
                               f" pointe sur un binaire inexistant: {p}")
        log.info("llama-server (%s, variable d'environnement): %s", flavor, p)
        return p

    root = _FLAVORS[flavor]["dir"]
    root.mkdir(parents=True, exist_ok=True)

    existing = _find_server_binary(root)
    if flavor == "prism":
        # Plusieurs builds du fork peuvent cohabiter (CUDA, ROCm…) : prendre le bon.
        pk = _prism_platform_key(_platform_key())
        existing = next((b for d in sorted(root.glob(f"*-{pk}"))
                         if (b := _find_server_binary(d))), None)
    if existing:
        log.info("llama-server binary found: %s", existing)
        return existing

    if flavor == "prism":
        log.info("Modèle PrismML (poids ternaires) : téléchargement du fork llama.cpp PrismML…")
    else:
        log.info("Downloading llama-server binary…")
    _download_release_binary(flavor)

    ready = _find_server_binary(root)
    if flavor == "prism":
        ready = next((b for d in sorted(root.glob(f"*-{pk}"))
                      if (b := _find_server_binary(d))), None)
    if ready:
        ready.chmod(0o755)
        log.info("llama-server ready: %s", ready)
        return ready

    raise RuntimeError(
        f"llama-server binary not found after download in {BINARY_DIR}. "
        "Download manually from https://github.com/ggml-org/llama.cpp/releases "
        f"and place in {BINARY_DIR}"
    )


def _prism_platform_key(pk: str) -> str:
    """Build du fork PrismML : jamais Vulkan quand mieux existe.

    Mesuré (Bonsai 2 27B PQ2_0) : noyaux Vulkan du ternaire pas au point — 0.9 tok/s
    en Vulkan, contre 74 en CUDA sur une 3090 (2026-09-23) et 53.5 en ROCm sur une
    RX 7900 XT (2026-09-24). VRM_PRISM_BACKEND=cuda|rocm|vulkan|cpu impose un choix.
    """
    forced = os.environ.get("VRM_PRISM_BACKEND")
    if forced:
        return f"linux-{forced}" if platform.system().lower() == "linux" else pk
    if pk != "linux-vulkan":
        return pk
    if _has_nvidia():
        return "linux-cuda"
    try:
        from core.rocm_runtime import rocm_usable
        if rocm_usable():
            return "linux-rocm"
    except Exception:
        log.debug("détection ROCm impossible", exc_info=True)
    return pk


def _download_release_binary(flavor: str = "upstream"):
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

    fl = _FLAVORS[flavor]
    root = fl["dir"]
    root.mkdir(parents=True, exist_ok=True)
    pk = _platform_key()
    if flavor == "prism":
        pk = _prism_platform_key(pk)
    if pk not in fl["assets"]:
        pk = "linux-cpu"
    tag = _latest_build_tag(flavor, pk if flavor != "upstream" else None)
    asset = fl["assets"][pk].format(tag=tag)
    url = f"https://github.com/{fl['repo']}/releases/download/{tag}/{asset}"

    log.info("Downloading %s (%s) …", url, pk)
    resp = _requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    data = resp.content

    dest_root = root / f"{tag}-{pk}"
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

    found = _find_server_binary(root)
    if found:
        found.chmod(0o755)
        for sibling in found.parent.glob("llama-*"):
            try:
                sibling.chmod(0o755)
            except Exception:
                pass
    log.info("llama.cpp %s (%s) extrait dans %s", tag, pk, dest_root)


def _find_server_binary(root: Optional[Path] = None):
    """Cherche llama-server dans `root` (récursif : les archives ont un bin/)."""
    root = root or BINARY_DIR
    for name in ("llama-server", "llama-server.exe", "server", "server.exe"):
        direct = root / name
        if direct.is_file():
            return direct
    for cand in sorted(root.rglob("llama-server*")):
        if cand.is_file():
            return cand
    return None


def _runtime_env(binary) -> dict:
    """Env d'exécution : les libggml-*.so vivent à côté du binaire, pas dans /usr/lib."""
    env = dict(os.environ)
    dirs = [str(Path(binary).parent)]
    if (Path(binary).parent / "libggml-hip.so").exists():
        # Build ROCm (fork PrismML sur AMD) : runtime ROCm 7, installé à la demande.
        from core.rocm_runtime import ensure_rocm_runtime
        dirs += ensure_rocm_runtime()
    if env.get("LD_LIBRARY_PATH"):
        dirs.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs)
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


def _compat_flags(binary, model_path=None) -> List[str]:
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
    if model_path and _too_big_for_ram(model_path):
        # Plus gros que la RAM disponible : sans mmap, le noyau tue le processus (OOM).
        # Avec mmap, les poids hors VRAM restent sur disque et sont relus à la demande :
        # lent mais ça répond (DeepSeek-V4-Flash, RAM plafonnée : cf. rapport 2026-09-23).
        if "--load-mode" in h:
            flags += ["--load-mode", "mmap"]
        log.warning("Modèle plus gros que la RAM disponible : chargement en mmap, les "
                    "poids hors VRAM seront relus depuis le disque (lent, mais sans planter)")
    elif "--load-mode" in h:
        flags += ["--load-mode", "none"]
    elif "--no-mmap" in h or not h:
        flags += ["--no-mmap"]
    if "--log-disable" in h:
        flags += ["--log-disable"]
    flags += _spec_flags(h, model_path)
    return flags


def _too_big_for_ram(model_path, reserve_gib: float = 8.0) -> bool:
    """Le modèle dépasse-t-il la RAM disponible (MemAvailable − réserve) ?"""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / 2 ** 20
                return _model_size_gib(model_path) > avail - reserve_gib
    except (OSError, ValueError):
        pass
    return False


def _model_size_gib(path) -> float:
    """Taille totale du GGUF, tous fragments « -0000k-of-0000n » compris."""
    import re
    p = Path(path)
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p.name)
    files = ([p.with_name(p.name[:m.start()] + f"-{k:05d}-of-{m.group(2)}.gguf")
              for k in range(1, int(m.group(2)) + 1)] if m else [p])
    return sum(f.stat().st_size for f in files if f.exists()) / 2 ** 30


def _gguf_header(path, tensor_types: bool = False, tensors: bool = False) -> Optional[dict]:
    """Lit l'en-tête GGUF SANS dépendance. None si le fichier n'est pas lisible
    (on ne devine jamais).

    Renvoie {"expert_count", "kv"} (métadonnées scalaires et chaînes courtes), plus
    "tensor_types" et/ou "tensors" [(nom, type, octets)] à la demande. La taille d'un
    tenseur se déduit des décalages de données : ça marche pour TOUT type, y compris
    ceux que le paquet `gguf` ne connaît pas (types ternaires PrismML 142, 143, sur
    lesquels il lève ValueError) — et sans ses 10-15 s d'indexation.
    """
    import struct as _st
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
              10: "Q", 11: "q", 12: "d"}
    import contextlib
    try:
        # `path` peut être un objet fichier (lecture distante par plages HTTP, core/predict.py)
        opened = contextlib.nullcontext(path) if hasattr(path, "read") else open(path, "rb")
        with opened as f:
            if f.read(4) != b"GGUF":
                return None
            version = _st.unpack("<I", f.read(4))[0]
            if version < 2:
                return None
            n_tensors, n_kv = _st.unpack("<QQ", f.read(16))

            def rd_str():
                n = _st.unpack("<Q", f.read(8))[0]
                return f.read(n)

            def skip_value(t):
                if t in scalar:
                    f.seek(_st.calcsize("<" + scalar[t]), 1)
                elif t == 8:
                    n = _st.unpack("<Q", f.read(8))[0]
                    f.seek(n, 1)
                elif t == 9:
                    it, n = _st.unpack("<IQ", f.read(12))
                    if it in scalar:
                        f.seek(_st.calcsize("<" + scalar[it]) * n, 1)
                    else:
                        for _ in range(n):
                            skip_value(it)
                else:
                    raise ValueError(f"type GGUF inconnu {t}")

            kv = {}
            for _ in range(n_kv):
                key = rd_str().decode("utf-8", "replace")
                t = _st.unpack("<I", f.read(4))[0]
                if t in scalar:
                    fmt = "<" + scalar[t]
                    kv[key] = _st.unpack(fmt, f.read(_st.calcsize(fmt)))[0]
                elif t == 8:
                    kv[key] = rd_str().decode("utf-8", "replace")[:256]
                else:
                    skip_value(t)
            out = {"kv": kv, "expert_count": next(
                (int(v) for k, v in kv.items() if k.endswith(".expert_count")), 0)}
            if not (tensor_types or tensors):
                return out
            infos = []
            for _ in range(n_tensors):
                name = rd_str().decode("utf-8", "replace")
                n_dims = _st.unpack("<I", f.read(4))[0]
                f.seek(8 * n_dims, 1)                                  # dimensions
                ttype, off = _st.unpack("<IQ", f.read(12))
                infos.append((name, ttype, off))
            out["tensor_types"] = {t for _, t, _ in infos}
            if tensors:
                align = int(kv.get("general.alignment", 32))
                data_start = -(-f.tell() // align) * align
                size = getattr(f, "size", None) or os.fstat(f.fileno()).st_size
                data_len = size - data_start
                ends = sorted({off for _, _, off in infos} | {data_len})
                nxt = {o: ends[i + 1] for i, o in enumerate(ends[:-1])}
                out["tensors"] = [(n, t, nxt[o] - o) for n, t, o in infos]
            return out
    except Exception:
        return None


def gguf_expert_count(path) -> Optional[int]:
    """Nombre d'experts d'un GGUF (0 = modèle dense), en ne lisant QUE l'en-tête."""
    h = _gguf_header(path)
    return None if h is None else h["expert_count"]


def needs_prism_fork(path) -> bool:
    """Le GGUF contient-il des tenseurs ternaires PrismML (Bonsai) ?"""
    h = _gguf_header(path, tensor_types=True)
    return bool(h and h.get("tensor_types", set()) & PRISM_TENSOR_TYPES)


def kv_bytes_per_token(path) -> Optional[int]:
    """Octets de cache KV par token en f16, d'après l'en-tête (None si inconnu).

    Modèles hybrides (Qwen3.5/3.6 : 1 couche d'attention sur 4, le reste en SSM) :
    seules les couches d'attention pleine ont un cache KV.
    """
    h = _gguf_header(path)
    if not h:
        return None
    kv = h["kv"]
    arch = kv.get("general.architecture")
    try:
        n_layer = int(kv[f"{arch}.block_count"])
        n_head = int(kv[f"{arch}.attention.head_count"])
        n_kv = int(kv.get(f"{arch}.attention.head_count_kv", n_head))
        d = int(kv[f"{arch}.embedding_length"]) // n_head
        kl = int(kv.get(f"{arch}.attention.key_length", d))
        vl = int(kv.get(f"{arch}.attention.value_length", d))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    interval = int(kv.get(f"{arch}.full_attention_interval", 1) or 1)
    return -(-n_layer // interval) * n_kv * (kl + vl) * 2


# Octets par élément relatifs à f16 (blocs de 32 : q8_0 = 34 o, q4_0 = 18 o).
_KV_TYPES = (("f16", 1.0), ("q8_0", 34 / 64), ("q4_0", 18 / 64))


def kv_cache_flags(model_path, n_ctx: int, devices: List[dict], help_text: str = "") -> List[str]:
    """Type du cache KV : f16 si le contexte tient, sinon q8_0, sinon q4_0 (VRM_KV_TYPE=auto).

    Mesuré le 2026-09-23 sur Qwen2.5-Coder-32B Q4_K_M, une 3090 seule : f16 ne tient
    pas à 24K tokens, q8_0 oui (26.9 tok/s), q4_0 tient à 48K (20.2 tok/s). Perplexité
    wikitext-2 : f16 6.210, q8_0 6.212 (+0.03 %), q4_0 6.225 (+0.2 %) ; débit −4 à −6 %.
    Quantifier plutôt que planter. Seulement si le modèle tient en VRAM : au-delà, c'est
    `vramancer plan` qui a mesuré la place réellement disponible.
    VRM_KV_TYPE=f16|q8_0|q4_0 impose un type.
    """
    want = os.environ.get("VRM_KV_TYPE", "auto")
    if help_text and "--cache-type-k" not in help_text:
        return []
    if want != "auto":
        return [] if want == "f16" else ["--cache-type-k", want, "--cache-type-v", want]
    per_tok = kv_bytes_per_token(model_path)
    free = sum(d.get("free_mib", 0) for d in devices if d.get("rpc") is not True) * 2 ** 20
    if not per_tok or not free:
        return []
    size = _model_size_gib(model_path) * 2 ** 30
    headroom = free - size - 2 ** 30 * max(1, len(devices))    # tampons de calcul
    if size > free or headroom <= 0:
        return []
    for name, ratio in _KV_TYPES:
        if per_tok * ratio * n_ctx <= headroom:
            if name != "f16":
                log.info("Cache KV en %s : %d tokens de contexte en f16 (%.1f Gio) dépassent "
                         "la VRAM restante (%.1f Gio). Perte mesurée ≤ 0.2 %% de perplexité.",
                         name, n_ctx, per_tok * n_ctx / 2 ** 30, headroom / 2 ** 30)
                return ["--cache-type-k", name, "--cache-type-v", name]
            return []
    log.warning("Contexte de %d tokens trop long même en q4_0 : réduire VRM_N_CTX", n_ctx)
    return ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]


def _spec_flags(help_text: str, model_path=None) -> List[str]:
    """Décodage spéculatif par n-grammes (prompt-lookup), piloté par VRM_SPEC.

    Mesuré le 2026-09-23 sur une tâche d'agent « réécris ce fichier en renommant une
    fonction » (décodage EXACT : même sortie que le modèle seul) :
      - modèle DENSE, Qwen2.5-Coder-32B Q4 (3090) : 36.2 → 282.9 tok/s (7.8x) ;
      - modèle DENSE, Qwen2.5-Coder-32B Q6_K réparti sur 3090 + 7900 XT :
        25.8 → 192.3 tok/s (7.5x), édition 73.9 s → 11.6 s ;
      - modèle MoE, Qwen3.6-35B-A3B sur la même paire : 99.3 → 34.6 tok/s (−65 %)
        alors que 93 % des tokens proposés étaient acceptés. Vérifier un lot de
        tokens active les experts de CHAQUE token : pour un MoE la vérification
        n'est pas « presque gratuite », et la spéculation coûte au lieu de rapporter.

    VRM_SPEC : "auto" (défaut du profil coding : n-grammes si le modèle est DENSE,
    rien si MoE ou illisible) · "ngram" (forcé) · "off" · ou un type llama.cpp brut.
    """
    spec = os.environ.get("VRM_SPEC", "off").strip().lower()
    if spec in ("", "off", "0", "none", "false") or "--spec-type" not in help_text:
        return []
    if spec == "auto":
        experts = gguf_expert_count(model_path) if model_path else None
        if experts != 0:
            log.info("Prompt-lookup désactivé : %s", "modèle MoE (%s experts) — mesuré "
                     "−65 %% sur MoE" % experts if experts else "type de modèle inconnu")
            return []
        log.info("Prompt-lookup ngram activé (modèle dense — mesuré 7.5x sur les éditions)")
        spec = "ngram"
    kind = "ngram-simple" if spec in ("ngram", "on", "1", "true") else spec
    if kind not in help_text:
        log.warning("VRM_SPEC=%s : type inconnu de ce llama-server, ignoré", kind)
        return []
    out = ["--spec-type", kind]
    n_max = os.environ.get("VRM_SPEC_N_MAX")
    if n_max and "--spec-draft-n-max" in help_text:
        out += ["--spec-draft-n-max", n_max]
    return out


def _free_port(preferred: int, tries: int = 20) -> int:
    """Premier port libre à partir de `preferred` (un orphelin peut squatter)."""
    for offset in range(tries):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                if offset:
                    log.warning("Port %d occupé (llama-server orphelin ?) — bascule sur %d",
                                preferred, port)
                return port
    raise RuntimeError(
        f"Aucun port libre entre {preferred} et {preferred + tries - 1}. "
        "Un llama-server orphelin tourne peut-être encore : `pkill -f llama-server`."
    )


def _die_with_parent():
    """Linux : demander au noyau de tuer ce processus si son parent meurt.

    `atexit` ne s'exécute pas si le parent est tué par SIGKILL — or un
    llama-server orphelin garde ~20 GB de VRAM. PR_SET_PDEATHSIG (1) couvre ce
    cas au niveau du noyau. No-op ailleurs que sur Linux.
    """
    if platform.system().lower() != "linux":
        return
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 15, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGTERM
    except Exception:
        pass


def _terminate_proc(proc) -> None:
    """Tue le sous-processus (appelé par le finalizer, sans référence à self)."""
    try:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


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

        binary = Path(binary_path) if binary_path else get_or_download_binary(model_path)

        # Un llama-server orphelin (parent tué sans shutdown) garde le port ET la
        # VRAM : le démarrage suivant échouait alors de façon incompréhensible
        # (« Address already in use », puis modèle non chargé). Constaté le
        # 2026-09-22. On cherche donc un port libre et on garantit le nettoyage.
        server_port = _free_port(server_port)
        self._port = server_port
        self._base_url = f"http://127.0.0.1:{server_port}"

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
        cmd += _compat_flags(binary, model_path)

        # Combien de GPU sont RÉELLEMENT disponibles ? torch.cuda ne compte que les
        # cartes NVIDIA : sur une machine mixte, l'appelant passe num_local_gpus=1
        # et on perdrait la carte AMD — donc le facteur 4.4x mesuré sur un modèle
        # qui ne tient que dans la VRAM cumulée. Le binaire, lui, les voit toutes.
        seen = backend_devices(binary, rpc_hosts=self._rpc_hosts)
        if len(seen) > max(1, num_local_gpus):
            log.info("%d devices vus par llama.cpp (%s) — on les utilise tous",
                     len(seen), ", ".join(d["name"] for d in seen))
            num_local_gpus = len(seen)
        cmd += kv_cache_flags(model_path, n_ctx, seen, _server_help(binary))

        # Répartition entre GPU : d'abord un split MESURÉ (vramancer tune-split),
        # sinon le prorata VRAM. Le prorata laisse jusqu'à 16 % sur la table quand
        # les cartes n'ont pas la même vitesse (mesuré 3090 + 7900 XT, 2026-09-23).
        if num_local_gpus > 1:
            split = None
            try:
                from core.split_tuner import cached_split
                split = cached_split(model_path, seen, n_ctx)
                if split:
                    log.info("Split mesuré (cache tune-split) : %s", split)
            except Exception:
                log.debug("cache tune-split illisible", exc_info=True)
            if not split and len(seen) > 1:
                # prorata VRAM, dans l'ordre de `seen` (RPC en tête, cf. backend_devices)
                split = [round(d["total_mib"] / 1024, 1) for d in seen]
            if not split:
                split = _local_tensor_split(num_local_gpus, binary=binary)
                if split and len(seen) > 1:
                    log.info("Split au prorata VRAM %s — `vramancer tune-split %s` "
                             "peut trouver mieux", split, Path(model_path).name)
            if split:
                cmd += ["--tensor-split", ",".join(str(s) for s in split)]

        # RPC remote nodes
        if self._rpc_hosts:
            cmd += ["--rpc", ",".join(self._rpc_hosts)]
            log.info("RPC nodes: %s", self._rpc_hosts)

        env = _runtime_env(binary)

        # Placement MESURÉ par `vramancer plan` (répartition des couches, ou chaud sur le
        # GPU principal + experts par étages) : prioritaire s'il existe pour ce modèle.
        # Promesse « ne jamais planter » : si llama-server refuse de démarrer avec le plan
        # (contexte bien plus long que celui des mesures, VRAM prise entre-temps…), on
        # retombe sur la répartition par défaut au lieu d'échouer.
        attempts = [cmd]
        if not self._rpc_hosts:
            try:
                from core.planner import cached_plan_args
                plan_args = cached_plan_args(model_path)
            except Exception:
                plan_args = None
            if plan_args:
                base = []
                skip = 0
                for a in cmd:
                    if skip:
                        skip -= 1
                        continue
                    if a in ("--n-gpu-layers", "--tensor-split"):
                        skip = 1
                        continue
                    base.append(a)
                # llama-bench sépare les règles -ot par « ; », llama-server par « , »
                plan_args = [a.replace(";", ",") if j and plan_args[j - 1] in ("-ot", "--override-tensor")
                             else a for j, a in enumerate(plan_args)]
                attempts = [base + plan_args, cmd]
                log.info("Placement mesuré (vramancer plan) : %s", " ".join(plan_args)[:160])

        # MoE : dernier recours tous experts en RAM (le chaud seul tient toujours en VRAM),
        # au lieu d'un -ngl -1 qui ne rentre pas (mesuré : DeepSeek 81 GiB → OOM).
        try:
            if gguf_expert_count(model_path) and "--cpu-moe" not in cmd:
                attempts.append(cmd + ["--cpu-moe"])
        except Exception:
            pass
        size_gib = _model_size_gib(model_path)
        ready_timeout = int(max(120, 8 * size_gib))    # ~650 s pour 81 GiB lus depuis le disque

        for i, attempt in enumerate(attempts):
            log.info("Starting llama-server: %s", " ".join(attempt[:6]) + " …")
            # stderr vers un fichier, pas un PIPE jamais lu : au-delà de 64 Ko de logs
            # llama-server se bloquerait sur write().
            self._stderr = tempfile.TemporaryFile()
            self._proc = subprocess.Popen(
                attempt,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
                env=env,
                preexec_fn=_die_with_parent if os.name == "posix" else None,
            )
            try:
                self._wait_ready(ready_timeout)
                break
            except RuntimeError as e:
                _terminate_proc(self._proc)
                if i == len(attempts) - 1:
                    raise
                log.warning("llama-server ne démarre pas (%s) — essai %d/%d avec un "
                            "placement plus prudent", str(e)[-160:], i + 2, len(attempts))
        # Nettoyage même si l'appelant oublie shutdown() ou meurt : sans ça, le
        # sous-processus survit et squatte ~20 GB de VRAM.
        self._finalizer = weakref.finalize(self, _terminate_proc, self._proc)
        atexit.register(self.shutdown)

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
                err = ""
                if getattr(self, "_stderr", None):
                    self._stderr.seek(0)
                    err = self._stderr.read().decode("utf-8", "ignore")[-800:]
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
        _terminate_proc(self._proc)
        self._proc = None
        fin = getattr(self, "_finalizer", None)
        if fin is not None:
            fin.detach()
        gc.collect()


# ── Helpers ───────────────────────────────────────────────────────────────────

def backend_devices(binary, rpc_hosts: Optional[List[str]] = None) -> List[dict]:
    """Devices tels que le BINAIRE les voit (`--list-devices`), dans SON ordre.

    C'est la seule source qui fasse autorité : torch.cuda ne voit que les cartes
    NVIDIA, donc sur une machine mixte il manquerait la carte AMD — et avec elle
    tout l'intérêt de la paire (mesuré le 2026-09-22 : 20.7 tok/s sur la 3090
    seule contre 91.7 tok/s sur la paire, pour un modèle de 27 GB qui ne tient
    dans aucune des deux cartes prises isolément).
    """
    out: List[dict] = []
    cmd = [str(binary)]
    if rpc_hosts:
        cmd += ["--rpc", ",".join(rpc_hosts)]
    cmd += ["--list-devices"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60, env=_runtime_env(binary))
        text = (r.stdout + r.stderr).decode("utf-8", "ignore")
    except Exception as e:
        log.debug("--list-devices indisponible: %s", e)
        return out
    import re
    for m in re.finditer(r"^\s*(\w+\d+):\s*(.+?)\s*\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB free)?\)",
                         text, re.MULTILINE):
        out.append({"id": m.group(1), "name": m.group(2),
                    "total_mib": int(m.group(3)),
                    "free_mib": int(m.group(4)) if m.group(4) else None,
                    "rpc": m.group(1).upper().startswith("RPC")})
    # ORDRE : `--list-devices` affiche les GPU locaux d'abord, mais `--tensor-split`
    # suit l'ordre interne de llama.cpp, où les périphériques RPC viennent EN TÊTE.
    # Vérifié le 2026-09-23 (b11112) en mesurant la VRAM : `-ts 78/22` avec une
    # 3090 locale + une 7900 XT en RPC ne laissait que 6.6 GB sur la 3090 — les 78 %
    # partaient sur la carte distante, qui débordait (12 tok/s au lieu de 98).
    out.sort(key=lambda d: 0 if d["rpc"] else 1)
    return out


def _local_tensor_split(num_gpus: int, binary=None) -> Optional[List[float]]:
    """Répartition proportionnelle à la VRAM, sur les devices RÉELLEMENT vus.

    Priorité au binaire (il voit NVIDIA *et* AMD) ; repli sur torch.cuda.
    """
    if binary is not None:
        devs = backend_devices(binary)
        if len(devs) > 1:
            return [round(d["total_mib"] / 1024, 1) for d in devs[:num_gpus or len(devs)]]
    try:
        import torch
        return [
            round(torch.cuda.mem_get_info(i)[1] / 1e9, 1)
            for i in range(min(num_gpus, torch.cuda.device_count()))
        ]
    except Exception:
        return None
