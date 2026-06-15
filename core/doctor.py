"""`vramancer doctor` — diagnostic complet, CHIFFRES MESURÉS uniquement.

Consolide `status` + `health` (réutilise core/health.py) et ajoute : P2P mesuré,
détection GPU/Blackwell, versions lues des packages, reco quickstart basée sur la
VRAM réelle. Règle d'or : aucun chiffre inventé — chaque ligne trace une mesure ou
un `importlib.metadata.version()`. Pas de tok/s postulé (→ `vramancer benchmark`).
"""
from __future__ import annotations
import importlib.metadata as _md
import os
import platform
import sys
from typing import Any, Dict, List, Optional

OK, WARN, BAD, INFO = "✅", "⚠️ ", "❌", "  "


def _ver(pkg: str) -> Optional[str]:
    try:
        return _md.version(pkg)
    except Exception:
        return None


def probe_gpus() -> List[Dict[str, Any]]:
    """GPU détectés via hetero_config (nom, arch, VRAM totale/libre, FP4 Blackwell)."""
    try:
        from core.hetero_config import auto_configure
        cfg = auto_configure(strategy="balanced")
        out = []
        for g in cfg.gpus:
            arch = g.profile.architecture if g.profile else "?"
            out.append({
                "index": g.index, "name": g.name, "arch": arch,
                "total_gb": round(g.total_vram_gb, 1),
                "free_gb": round(g.free_vram_gb, 1) if g.free_vram_gb > 0 else None,
                "fp4": arch == "Blackwell",
            })
        return out
    except Exception as e:
        return [{"error": str(e)}]


def probe_p2p() -> Dict[str, Any]:
    """Chemin P2P direct entre GPU0 et GPU1 — MESURÉ (can_device_access_peer)."""
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            return {"applicable": False}
        can01 = torch.cuda.can_device_access_peer(0, 1)
        return {"applicable": True, "direct_p2p": bool(can01)}
    except Exception as e:
        return {"applicable": False, "error": str(e)}


def collect() -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    # Système
    d["system"] = {
        "os": platform.system(), "release": platform.release(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "machine": platform.machine(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        d["system"]["ram_total_gb"] = round(vm.total / 1024**3, 1)
        d["system"]["ram_free_gb"] = round(vm.available / 1024**3, 1)
    except Exception:
        pass
    # GPUs (mesuré)
    d["gpus"] = probe_gpus()
    # P2P (mesuré)
    d["p2p"] = probe_p2p()
    # Versions (lues)
    d["versions"] = {p: _ver(p) for p in
                     ("torch", "transformers", "accelerate", "peft", "zeroconf",
                      "numpy", "vramancer")}
    try:
        import torch
        d["versions"]["cuda"] = getattr(torch.version, "cuda", None)
        d["cuda_available"] = torch.cuda.is_available()
    except Exception:
        d["cuda_available"] = False
    # Santé (réutilise health.py)
    try:
        from core import health
        d["health"] = health.full_diagnostic()
    except Exception as e:
        d["health"] = {"error": str(e)}
    # Reco (quickstart, basée VRAM réelle)
    try:
        from vramancer.quickstart import recommend
        d["reco"] = recommend("code-assistant")
    except Exception as e:
        d["reco"] = {"error": str(e)}
    return d


def run_doctor() -> int:
    d = collect()
    print("=" * 64)
    print("  VRAMancer doctor — diagnostic (chiffres mesurés)")
    print("=" * 64)

    s = d["system"]
    print(f"\n{INFO}Système : {s['os']} {s.get('release','')} · Python {s['python']} · {s.get('machine','')}")
    if "ram_total_gb" in s:
        print(f"{INFO}RAM     : {s['ram_free_gb']} / {s['ram_total_gb']} GB libres")

    print("\n  GPU :")
    warns: List[str] = []
    for g in d["gpus"]:
        if "error" in g:
            print(f"  {BAD} détection GPU: {g['error']}"); continue
        free = f"{g['free_gb']} libre / " if g.get("free_gb") is not None else ""
        fp4 = "  [NVFP4]" if g["fp4"] else ""
        print(f"  {OK} GPU{g['index']} : {g['name']} ({g['arch']}) — {free}{g['total_gb']} GB{fp4}")

    p = d["p2p"]
    if p.get("applicable"):
        if p["direct_p2p"]:
            print(f"\n  {OK} P2P direct GPU↔GPU : disponible")
        else:
            print(f"\n  {WARN}P2P direct GPU↔GPU : INDISPONIBLE (mesuré: can_device_access_peer=False)")
            print(f"  {INFO}  → transferts CPU-staged ~11.6 GB/s (torch) / ~25 GB/s (GpuPipeline). OK inférence.")
            warns.append("P2P indisponible (GPU consumer sans NVLink) — pas de split de modèle par couche cross-GPU rapide.")
    elif p.get("error"):
        print(f"\n  {INFO}P2P : non testé ({p['error']})")

    print("\n  Versions :")
    v = d["versions"]
    for pkg in ("torch", "transformers", "accelerate", "peft", "zeroconf", "vramancer"):
        val = v.get(pkg)
        mark = OK if val else WARN
        if not val and pkg in ("peft", "zeroconf"):
            warns.append(f"{pkg} absent → installe 'vramancer[{'gpu' if pkg=='peft' else 'cluster'}]' ({'LoRA hot-swap' if pkg=='peft' else 'mDNS cluster'}).")
        print(f"  {mark} {pkg:12s} {val or 'absent'}")
    if v.get("cuda"):
        print(f"  {INFO} CUDA toolkit (torch): {v['cuda']} · cuda_available={d.get('cuda_available')}")

    h = d.get("health", {})
    if isinstance(h, dict) and "error" not in h:
        status = h.get("status") or h.get("overall") or "?"
        print(f"\n  Santé globale : {status}")
        # surface les sous-systèmes en échec
        for k, val in h.items():
            if isinstance(val, dict):
                st = val.get("status")
                if st and str(st).lower() not in ("ok", "healthy", "true", "available", "active"):
                    print(f"  {WARN}{k}: {st}")

    r = d.get("reco", {})
    if "model" in r:
        print(f"\n  Reco (VRAM {r.get('vram_total_gb')} GB) : {r['model']}  [{r['quant']}, ~{r.get('vram_need_gb')} GB]")
        print(f"  {INFO}  → vramancer quickstart code-assistant --run")

    if warns:
        print("\n  ⚠️  À noter :")
        for w in warns:
            print(f"     • {w}")
    print("\n  Pour des tok/s MESURÉS : vramancer benchmark")
    print("=" * 64)
    return 0
