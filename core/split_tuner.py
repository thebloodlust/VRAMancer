"""Réglage MESURÉ du partage des couches entre GPU (`vramancer tune-split`).

Pourquoi : la répartition « au prorata de la VRAM » (ce que faisait VRAMancer, et le
défaut de llama.cpp) n'est pas la bonne dès que les GPU n'ont pas la même vitesse.
En décodage mono-flux, les couches s'exécutent l'une après l'autre : le temps par
token est la SOMME des temps de chaque carte, donc il faut mettre le plus de couches
possible sur la carte la plus rapide, et seulement le reste sur l'autre.

Mesuré le 2026-09-23 sur RTX 3090 + RX 7900 XT, Qwen3.6-35B-A3B Q6_K (27.3 GB) :

    part sur l'AMD   45 % (∝ VRAM)   30 %    20 %    16 %    12 %
    tok/s            91.4            98.1    102.8   106.4   échec (3090 pleine)

→ +16 % sans rien changer au matériel. Mais QUELLE carte est la plus rapide ne se
devine pas (les bandes passantes théoriques 3090/7900 XT ne diffèrent que de 17 %,
l'écart réel vient surtout des pilotes). D'où un réglage mesuré : quelques passes
de `llama-bench` sur le vrai modèle, à la vraie profondeur de contexte (sinon le
cache KV du serveur ferait déborder la carte remplie), résultat mis en cache et
réutilisé automatiquement par `vramancer serve`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("vramancer.split_tuner")

CACHE_FILE = Path.home() / ".cache" / "vramancer" / "splits.json"


# ── Cache ─────────────────────────────────────────────────────────────────────

def _key(model_path: str, devices: List[dict], n_ctx: int) -> str:
    """Clé = modèle (nom + taille, pas de hash de 27 GB) + GPU dans l'ordre + contexte."""
    p = Path(model_path)
    size = p.stat().st_size if p.exists() else 0
    names = "+".join(d["name"] for d in devices)
    return f"{p.name}|{size}|{names}|ctx{n_ctx}"


def _load_cache() -> Dict[str, dict]:
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_cache(cache: Dict[str, dict]) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    except Exception as e:
        log.warning("Impossible d'écrire le cache de split (%s)", e)


def cached_split(model_path: str, devices: List[dict], n_ctx: int) -> Optional[List[float]]:
    """Split déjà mesuré pour ce modèle, ces GPU et ce contexte, sinon None."""
    if len(devices) < 2:
        return None
    entry = _load_cache().get(_key(model_path, devices, n_ctx))
    return entry["split"] if entry else None


# ── Candidats ─────────────────────────────────────────────────────────────────

def _split_for(devices: List[dict], fast: int, share: float) -> List[float]:
    """`share` sur la carte `fast`, le reste réparti au prorata VRAM entre les autres."""
    total = [float(d["total_mib"]) for d in devices]
    others = [i for i in range(len(devices)) if i != fast]
    o_tot = sum(total[i] for i in others)
    split = [0.0] * len(devices)
    split[fast] = round(share, 4)
    for i in others:
        split[i] = round((1.0 - share) * total[i] / o_tot, 4)
    return split


def fill_cap(devices: List[dict], fast: int, model_bytes: int,
             reserve_mib: int = 1024) -> float:
    """Part maximale du modèle que la carte `fast` peut prendre (estimation)."""
    free = float(devices[fast].get("free_mib") or devices[fast]["total_mib"])
    return max(0.0, min(0.97, (free - reserve_mib) / (model_bytes / (1024 * 1024))))


def candidate_splits(devices: List[dict], model_bytes: int, fast: Optional[int] = None,
                     steps: int = 4, step: float = 0.04,
                     reserve_mib: int = 1024) -> List[List[float]]:
    """Prorata VRAM (référence) + remplissage maximal de la carte `fast` et quelques
    crans en retrait. Sans `fast` : le remplissage maximal de CHAQUE carte (pour
    découvrir laquelle est la rapide)."""
    n = len(devices)
    if n < 2 or model_bytes <= 0:
        return []
    total = [float(d["total_mib"]) for d in devices]
    out: List[List[float]] = [[round(t / sum(total), 4) for t in total]]
    fasts = range(n) if fast is None else [fast]
    for f in fasts:
        cap = fill_cap(devices, f, model_bytes, reserve_mib)
        for k in range(steps if fast is not None else 1):
            share = cap - k * step
            if share <= 0.05:
                break
            sp = _split_for(devices, f, share)
            if sp not in out:
                out.append(sp)
    return out


# ── Mesure ────────────────────────────────────────────────────────────────────

_PP_RE = re.compile(r"\|\s*pp\d+[^|]*\|\s*([\d.]+)\s*±")
_TG_RE = re.compile(r"\|\s*tg\d+[^|]*\|\s*([\d.]+)\s*±")


def _bench(bench_bin: Path, model_path: str, split: List[float], n_ctx: int,
           env: dict, reps: int = 2, timeout: int = 900,
           rpc_hosts: Optional[List[str]] = None) -> Optional[dict]:
    """Prefill ET génération pour un split, à la profondeur de contexte demandée.

    Mesurer le prefill n'est pas un luxe : sur Qwen3.6-35B Q8_0, remplir la 3090 à
    68 % donnait la MEILLEURE génération (96.8 tok/s) mais faisait s'effondrer le
    prefill de 2 726 à 720 tok/s (tampons de prefill qui ne tiennent plus). Un tuner
    aveugle au prefill aurait choisi ce split et rendu chaque long prompt 4x plus lent.

    Renvoie {"pp": tok/s, "tg": tok/s} ou None si le split ne tient pas en mémoire.
    """
    ts = "/".join(f"{s:.4f}" for s in split)
    cmd = [str(bench_bin), "-m", model_path, "-ngl", "99", "-fa", "on",
           "-p", "512", "-n", "64", "-d", str(n_ctx), "-r", str(reps),
           "-ts", ts, "-o", "md"]
    if rpc_hosts:
        cmd += ["--rpc", ",".join(rpc_hosts)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout.decode("utf-8", "ignore")
    pp, tg = _PP_RE.search(out), _TG_RE.search(out)
    if not tg:
        return None
    return {"pp": float(pp.group(1)) if pp else None, "tg": float(tg.group(1))}


# Un split dont le prefill tombe sous cette fraction du meilleur prefill observé
# est écarté, même s'il génère plus vite.
PP_FLOOR = 0.6


def tune(model_path: str, server_binary, n_ctx: int = 16384,
         report=print, rpc_hosts: Optional[List[str]] = None) -> Optional[dict]:
    """Mesure les candidats, garde le plus rapide qui TIENT, et le met en cache."""
    from core.llama_server_backend import backend_devices, _runtime_env

    server_binary = Path(server_binary)
    bench_bin = server_binary.parent / "llama-bench"
    if not bench_bin.exists():
        report(f"llama-bench introuvable à côté de {server_binary} — réglage impossible.")
        return None
    devices = backend_devices(server_binary, rpc_hosts=rpc_hosts)   # RPC en tête
    if len(devices) < 2:
        report("Un seul GPU visible : rien à répartir.")
        return None

    env = _runtime_env(server_binary)
    size = Path(model_path).stat().st_size
    report(f"{len(devices)} GPU : " + ", ".join(d['name'] for d in devices))
    report(f"Modèle {size / 2**30:.1f} GiB, contexte {n_ctx}")

    results: List[dict] = []
    seen_splits = []

    def measure(split: List[float]) -> Optional[float]:
        if split in seen_splits:
            return next(r["tok_s"] for r in results if r["split"] == split)
        seen_splits.append(split)
        m = _bench(bench_bin, model_path, split, n_ctx, env, rpc_hosts=rpc_hosts)
        label = " / ".join(f"{x * 100:.0f}%" for x in split)
        if m is None:
            report(f"  {label:<24} ne tient pas")
        else:
            pp = f"prefill {m['pp']:.0f}" if m.get("pp") else "prefill ?"
            report(f"  {label:<24} {m['tg']:.1f} tok/s  ({pp})")
        results.append({"split": split, "tok_s": m["tg"] if m else None,
                        "pp": m.get("pp") if m else None})
        return m["tg"] if m else None

    # 1. référence + remplissage maximal de chaque carte : laquelle est la rapide ?
    report("Étape 1 — quelle carte est la plus rapide ?")
    for sp in candidate_splits(devices, size):
        measure(sp)
    best_fast, best_val = None, -1.0
    for f in range(len(devices)):
        # premier split (en reculant) de la carte f qui tient en mémoire
        for sp in candidate_splits(devices, size, fast=f, steps=6)[1:]:
            v = measure(sp)
            if v:
                if v > best_val:
                    best_fast, best_val = f, v
                break
    if best_fast is None:
        report("Aucune répartition ne tient en mémoire à ce contexte.")
        return None

    # 2. affiner dans le bon sens (la limite de remplissage n'est qu'une estimation)
    report(f"Étape 2 — affinage autour de {devices[best_fast]['name']}")
    for sp in candidate_splits(devices, size, fast=best_fast, steps=4)[1:]:
        measure(sp)

    ok = [r for r in results if r["tok_s"]]
    if not ok:
        report("Aucune répartition ne tient en mémoire à ce contexte.")
        return None
    pps = [r["pp"] for r in ok if r.get("pp")]
    if pps:
        floor = PP_FLOOR * max(pps)
        rejected = [r for r in ok if r.get("pp") and r["pp"] < floor]
        for r in rejected:
            report(f"  écarté : {' / '.join(f'{x * 100:.0f}%' for x in r['split'])} — "
                   f"prefill {r['pp']:.0f} < {floor:.0f} (effondrement)")
        ok = [r for r in ok if r not in rejected] or ok
    best = max(ok, key=lambda r: r["tok_s"])
    base = results[0]["tok_s"]
    gain = f" ({(best['tok_s'] / base - 1) * 100:+.0f} % vs prorata VRAM)" if base else ""
    report(f"Retenu : {' / '.join(f'{s * 100:.0f}%' for s in best['split'])} → "
           f"{best['tok_s']:.1f} tok/s{gain}")

    cache = _load_cache()
    cache[_key(model_path, devices, n_ctx)] = {
        "split": best["split"], "tok_s": best["tok_s"],
        "baseline_tok_s": base, "all": results,
    }
    _save_cache(cache)
    return cache[_key(model_path, devices, n_ctx)]


__all__ = ["tune", "cached_split", "candidate_splits", "CACHE_FILE"]
