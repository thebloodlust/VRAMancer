"""Prédire AVANT de télécharger : tient ou pas, où vont les poids, quel débit.

On lit l'en-tête d'un GGUF distant par requêtes HTTP partielles (quelques Mo sur des
dizaines de Go), puis on applique le modèle de coût de
docs/reports/PLANIFICATEUR_COUT_2026-09-23.md :

    temps/token = Σ étages [octets lus par token sur l'étage / bande passante effective]
                + Σ GPU [couches sur ce GPU × coût fixe par couche]

Les octets lus par token : tout le modèle pour un dense ; pour un MoE, la partie chaude
(attention, normes, expert partagé) plus experts_actifs/experts de chaque couche.

C'est un ordre de grandeur, pas une mesure : ~2 % d'erreur médiane sur l'étage GPU
une fois calibré, bien plus grossier sur CPU et disque. Après téléchargement,
`vramancer plan` mesure pour de bon.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

GIB = 2 ** 30

# Paramètres effectifs mesurés le 2026-09-23 (VM Proxmox, llama.cpp b11112 Vulkan).
# Clé = sous-chaîne du nom du GPU. (bande passante Gio/s, coût fixe par couche en s)
GPU_PARAMS: Dict[str, Tuple[float, float]] = {
    "RTX 3090": (925.0, 122e-6),
    "7900 XT": (928.0, 248e-6),
}
UNKNOWN_GPU = (500.0, 200e-6)          # prudent, faute de mieux : à calibrer
RAM_GIBS = 58.0                        # EPYC 7402 AVX2, 16 vCPU, experts en RAM
DISK_GIBS = 1.8 * 1e9 / GIB            # lectures aléatoires de 4 Mo (experts dispersés)
MARGIN_GIB = 1.5                       # tampons de calcul + cache KV court, par GPU
# Architectures sur lesquelles les coûts par couche ont été calibrés et vérifiés
# (−9 à +16 % sur 5 cas). Ailleurs, l'attention peut coûter bien plus : DeepSeek-V4
# (attention compressée + indexeur) est surestimé ×2.4 à ×2.8 → borne haute seulement.
CALIBRATED_ARCHS = {"llama", "qwen2", "qwen3", "qwen3moe", "qwen35", "qwen35moe"}


# ── Lecture distante ────────────────────────────────────────────────────────

class HttpRangeFile:
    """Fichier distant en lecture seule, par blocs HTTP Range (mis en cache)."""

    def __init__(self, url: str, block: int = 1 << 20, timeout: int = 30):
        self._block, self._timeout, self._cache = block, timeout, {}
        self._s = requests.Session()
        r = self._s.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout,
                        allow_redirects=True)
        r.raise_for_status()
        m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
        if r.status_code != 206 or not m:
            raise RuntimeError("le serveur ne gère pas les requêtes partielles (Range)")
        self.url, self.size, self.pos = r.url, int(m.group(1)), 0
        self.fetched = 0

    def _get_block(self, i: int) -> bytes:
        if i not in self._cache:
            a = i * self._block
            b = min(a + self._block, self.size) - 1
            r = self._s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=self._timeout)
            r.raise_for_status()
            self._cache[i] = r.content
            self.fetched += len(r.content)
        return self._cache[i]

    def read(self, n: int) -> bytes:
        out = bytearray()
        while n > 0 and self.pos < self.size:
            blk = self._get_block(self.pos // self._block)
            off = self.pos % self._block
            chunk = blk[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def tell(self) -> int:
        return self.pos

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._s.close()


def hf_url(ref: str) -> str:
    """`org/depot/fichier.gguf` ou `hf://org/depot/fichier.gguf` → URL de téléchargement."""
    if ref.startswith(("http://", "https://")):
        return ref
    ref = ref.removeprefix("hf://")
    org, repo, path = ref.split("/", 2)
    return f"https://huggingface.co/{org}/{repo}/resolve/main/{path}"


def _shard_refs(ref: str) -> List[str]:
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", ref)
    if not m:
        return [ref]
    n = int(m.group(2))
    return [ref[:m.start()] + f"-{k:05d}-of-{m.group(2)}.gguf" for k in range(1, n + 1)]


@dataclass
class Profile:
    n_layers: int
    total: int                       # octets
    experts: int                     # octets d'experts routés (tous)
    expert_count: int = 0
    expert_used: int = 0
    fetched: int = 0                 # octets réellement téléchargés pour le savoir
    arch: str = ""

    @property
    def hot(self) -> int:
        return self.total - self.experts

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 0 and self.experts > 0

    @property
    def active_frac(self) -> float:
        return self.expert_used / self.expert_count if self.is_moe else 1.0


def remote_profile(ref: str) -> Profile:
    """Profil d'un GGUF distant (tous fragments) sans le télécharger."""
    from core.llama_server_backend import _gguf_header
    from core.planner import EXP_RE
    total = experts = fetched = 0
    n_layers = ne = nu = 0
    for i, r in enumerate(_shard_refs(ref)):
        with HttpRangeFile(hf_url(r)) as f:
            h = _gguf_header(f, tensors=True)
            fetched += f.fetched
        if h is None:
            raise ValueError(f"en-tête GGUF illisible : {r}")
        if i == 0:
            kv = h["kv"]
            arch = kv["general.architecture"]
            n_layers = int(kv[f"{arch}.block_count"])
            ne = int(kv.get(f"{arch}.expert_count", 0))
            nu = int(kv.get(f"{arch}.expert_used_count", 0))
        for name, _t, b in h["tensors"]:
            total += b
            if EXP_RE.match(name):
                experts += b
    return Profile(n_layers, total, experts, ne, nu, fetched, arch)


def local_profile(path: str) -> Profile:
    from core.planner import profile_model
    p = profile_model(path)
    from core.llama_server_backend import _gguf_header
    arch = (_gguf_header(path) or {}).get("kv", {}).get("general.architecture", "")
    return Profile(p.n_layers, p.total_bytes, sum(p.expert_bytes_per_layer.values()),
                   p.expert_count, p.expert_used, arch=arch)


# ── Prédiction ──────────────────────────────────────────────────────────────

@dataclass
class Tier:
    name: str
    capacity: float                  # Gio utilisables
    gibs: float                      # bande passante effective
    layer_cost: float = 0.0          # s par couche (GPU)
    is_gpu: bool = False


@dataclass
class Prediction:
    regime: str
    tok_s: Optional[float]
    placement: List[Tuple[str, float]] = field(default_factory=list)   # (étage, Gio)


def machine_tiers(devices: List[dict], ram_avail_gib: Optional[float] = None) -> List[Tier]:
    """Étages de CETTE machine, du plus rapide au plus lent."""
    tiers = []
    for d in devices:
        if d.get("rpc"):
            continue
        bw, c = next((v for k, v in GPU_PARAMS.items() if k in d["name"]), UNKNOWN_GPU)
        free = (d.get("free_mib") or d["total_mib"]) / 1024
        tiers.append(Tier(d["name"], max(0.0, free - MARGIN_GIB), bw, c, True))
    if ram_avail_gib is None:
        ram_avail_gib = _mem_available_gib()
    tiers.append(Tier("RAM", max(0.0, ram_avail_gib - 4.0), RAM_GIBS))
    tiers.append(Tier("disque (mmap)", float("inf"), DISK_GIBS))
    return tiers


def _mem_available_gib() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 8.0


def predict(p: Profile, tiers: List[Tier]) -> Prediction:
    """Remplit les étages du plus rapide au plus lent (le chaud d'abord), prédit tok/s."""
    gpus = [t for t in tiers if t.is_gpu]
    L = max(1, p.n_layers)
    placement: List[Tuple[str, float]] = []
    t = 0.0

    def cost(tier: Tier, gib: float, read_frac: float, layers: float):
        nonlocal t
        placement.append((tier.name, gib))
        t += gib * read_frac / tier.gibs + layers * tier.layer_cost

    if p.is_moe and gpus and p.hot / GIB <= gpus[0].capacity:
        # Chaud sur le GPU principal ; experts, lus à active_frac, sur les étages suivants.
        regime = "MoE : chaud sur le GPU principal, experts par étages"
        hot = p.hot / GIB
        per_layer = p.experts / GIB / L
        left = p.experts / GIB
        room = gpus[0].capacity - hot
        k = min(L, int(room // per_layer)) if per_layer else L
        placement.append((gpus[0].name, hot + k * per_layer))
        t += hot / gpus[0].gibs + k * per_layer * p.active_frac / gpus[0].gibs + L * gpus[0].layer_cost
        left -= k * per_layer
        for tier in tiers[1:]:
            if left <= 1e-9:
                break
            take = left if tier.capacity == float("inf") else min(
                left, (int(tier.capacity // per_layer) * per_layer) if tier.is_gpu else tier.capacity)
            if take <= 0:
                continue
            placement.append((tier.name, take))
            t += take * p.active_frac / tier.gibs
            if tier.is_gpu:
                t += 2 * gpus[0].layer_cost               # aller-retour des activations
            left -= take
        if all(n in {g.name for g in gpus} for n, _ in placement):
            regime = "tient en VRAM (MoE : chaud sur le GPU principal)"
    else:
        # Couches entières, GPU le moins cher d'abord, puis RAM (couches CPU), puis disque.
        left = p.total / GIB
        regime = "tient en VRAM" if left <= sum(g.capacity for g in gpus) else (
            "déborde en RAM" if left <= sum(x.capacity for x in tiers[:-1]) else
            "déborde sur disque (lent mais ne plante pas)")
        per_layer = left / L
        for tier in sorted(gpus, key=lambda g: g.layer_cost + per_layer / g.gibs) + tiers[len(gpus):]:
            if left <= 1e-9:
                break
            take = min(left, tier.capacity)
            if tier.is_gpu:
                take = int(take // per_layer) * per_layer
            if take <= 0:
                continue
            cost(tier, take, p.active_frac if p.is_moe else 1.0,
                 take / per_layer if tier.is_gpu else 0)
            left -= take
    return Prediction(regime, (1.0 / t) if t > 0 else None, placement)


def fmt_prediction(p: Profile, pred: Prediction) -> str:
    kind = (f"MoE {p.expert_used}/{p.expert_count} experts" if p.is_moe else "dense")
    lines = [f"Modèle {p.total / GIB:.1f} Gio, {p.n_layers} couches, {kind}"
             + (f" · en-tête lu : {p.fetched / 2 ** 20:.1f} Mo" if p.fetched else ""),
             f"Régime : {pred.regime}"]
    lines += [f"  {name:<28} {gib:6.1f} Gio" for name, gib in pred.placement]
    if pred.tok_s and p.arch in CALIBRATED_ARCHS:
        lines.append(f"Débit prédit : ~{pred.tok_s:.1f} tok/s (±15 % sur GPU, plus grossier "
                     "en RAM ; `vramancer plan` mesure après téléchargement)")
    elif pred.tok_s:
        lines.append(f"Débit : au plus ~{pred.tok_s:.1f} tok/s — architecture « {p.arch} » non "
                     "calibrée, le réel peut être 2 à 3× plus bas (DeepSeek-V4 : ×2.4 à ×2.8). "
                     "`vramancer plan` mesure après téléchargement.")
    return "\n".join(lines)


__all__ = ["HttpRangeFile", "remote_profile", "local_profile", "machine_tiers",
           "predict", "fmt_prediction", "Profile", "Tier", "Prediction", "hf_url"]
