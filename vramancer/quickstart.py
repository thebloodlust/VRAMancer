"""S2 — `vramancer quickstart <use-case>` : l'utilisateur choisit un USAGE, pas un modèle.

VRAMancer détecte le matériel et recommande le plus gros modèle adapté qui **tient**
dans la VRAM mesurée, avec la quantification adaptée au GPU (NVFP4 si Blackwell,
sinon NF4). Produit une commande prête (`vramancer run …`), ou la lance avec `--run`.

Honnête : pas de magie. Estimation VRAM transparente (params × octets/param × marge),
catalogue de modèles connus-bons par usage. Le but = supprimer le « quel modèle ? ».
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Candidate:
    model: str
    params_b: float          # milliards de paramètres
    note: str = ""


# Catalogue par usage : du plus gros au plus petit. On garde le 1er qui tient.
CATALOG = {
    "code-assistant": [
        Candidate("Qwen/Qwen2.5-Coder-32B-Instruct", 32, "code SOTA, gros"),
        Candidate("Qwen/Qwen2.5-Coder-14B-Instruct", 14, "excellent compromis code"),
        Candidate("Qwen/Qwen2.5-Coder-7B-Instruct", 7, "rapide, bon code"),
        Candidate("Qwen/Qwen2.5-Coder-3B-Instruct", 3, "léger"),
        Candidate("Qwen/Qwen2.5-Coder-1.5B-Instruct", 1.5, "edge/CPU"),
    ],
    "chat": [
        Candidate("Qwen/Qwen2.5-32B-Instruct", 32, "chat très capable"),
        Candidate("Qwen/Qwen2.5-14B-Instruct", 14, "compromis chat"),
        Candidate("meta-llama/Llama-3.1-8B-Instruct", 8, "populaire"),
        Candidate("Qwen/Qwen2.5-7B-Instruct", 7, "rapide"),
        Candidate("Qwen/Qwen2.5-3B-Instruct", 3, "léger"),
    ],
    "summarize": [
        Candidate("Qwen/Qwen2.5-14B-Instruct", 14, "synthèse longue"),
        Candidate("Qwen/Qwen2.5-7B-Instruct", 7, "rapide"),
        Candidate("Qwen/Qwen2.5-3B-Instruct", 3, "léger"),
    ],
}
USE_CASES = list(CATALOG.keys())

# Octets par paramètre selon la quantification.
_BPP = {"nvfp4": 0.60, "nf4": 0.55, "int8": 1.0, "bf16": 2.0}


def _detect():
    """Renvoie (vram_libre_totale_GB, blackwell_present, [descr GPU])."""
    try:
        from core.hetero_config import auto_configure
        cfg = auto_configure(strategy="balanced")
        gpus = cfg.gpus
        total = sum((g.free_vram_gb if g.free_vram_gb > 0 else g.total_vram_gb) for g in gpus)
        blackwell = any(
            (g.profile and g.profile.architecture == "Blackwell") for g in gpus
        )
        descr = [f"{g.name} ({(g.free_vram_gb or g.total_vram_gb):.0f} GB)" for g in gpus]
        return total, blackwell, descr
    except Exception as e:  # pragma: no cover
        return 0.0, False, [f"détection indisponible: {e}"]


def _vram_need_gb(params_b: float, quant: str) -> float:
    """Besoin VRAM estimé : poids + marge KV/activations + runtime fixe."""
    return params_b * _BPP[quant] * 1.20 + 1.5


def recommend(use_case: str) -> dict:
    """Choisit le plus gros modèle de l'usage qui tient dans la VRAM détectée."""
    if use_case not in CATALOG:
        return {"error": f"usage inconnu '{use_case}'. Choix: {', '.join(USE_CASES)}"}
    total, blackwell, descr = _detect()
    quant = "nvfp4" if blackwell else "nf4"  # quant adapté au matériel
    for c in CATALOG[use_case]:
        need = _vram_need_gb(c.params_b, quant)
        if need <= total:
            # bf16 possible si ça tient confortablement → meilleure qualité
            bf16_ok = _vram_need_gb(c.params_b, "bf16") <= total
            return {
                "use_case": use_case, "model": c.model, "params_b": c.params_b,
                "quant": "bf16" if bf16_ok else quant,
                "vram_need_gb": round(_vram_need_gb(c.params_b, "bf16" if bf16_ok else quant), 1),
                "vram_total_gb": round(total, 1), "gpus": descr, "note": c.note,
                "blackwell": blackwell,
            }
    # rien ne tient : proposer le plus petit en quant agressif + offload CPU
    smallest = CATALOG[use_case][-1]
    return {
        "use_case": use_case, "model": smallest.model, "params_b": smallest.params_b,
        "quant": quant, "vram_need_gb": round(_vram_need_gb(smallest.params_b, quant), 1),
        "vram_total_gb": round(total, 1), "gpus": descr, "note": smallest.note + " (offload CPU probable)",
        "blackwell": blackwell, "tight": True,
    }


def run_quickstart(use_case: str, launch: bool = False, serve: bool = False) -> int:
    rec = recommend(use_case)
    if "error" in rec:
        print(rec["error"])
        return 2
    print("=" * 60)
    print(f"  VRAMancer quickstart — usage : {rec['use_case']}")
    print("=" * 60)
    print(f"  Matériel détecté : {', '.join(rec['gpus'])}")
    print(f"  VRAM totale      : {rec['vram_total_gb']} GB")
    print(f"  → Modèle         : {rec['model']}  ({rec['params_b']}B, {rec['note']})")
    print(f"  → Quantification : {rec['quant']}" + ("  [NVFP4 Blackwell]" if rec.get("blackwell") and rec["quant"] == "nvfp4" else ""))
    print(f"  → VRAM estimée   : ~{rec['vram_need_gb']} GB")
    if rec.get("tight"):
        print("  ⚠ Aucun modèle de l'usage ne tient pleinement — offload CPU probable (plus lent).")
    cmd = f"vramancer {'serve' if serve else 'run'} {rec['model']}"
    if rec["quant"] != "bf16":
        cmd += f" -q {rec['quant']}" if not serve else f" --quantization {rec['quant']}"
    print("-" * 60)
    print(f"  Commande : {cmd}")
    print("=" * 60)
    if not launch:
        return 0
    # Lancement direct via le chemin existant.
    print("\n[quickstart] lancement…\n")
    from types import SimpleNamespace
    from vramancer.main import _cmd_run
    q = None if rec["quant"] == "bf16" else rec["quant"]
    _cmd_run(SimpleNamespace(
        model=rec["model"], prompt=None, max_tokens=256, temperature=0.7,
        backend="auto", gpus=None, quantization=q,
    ))
    return 0
