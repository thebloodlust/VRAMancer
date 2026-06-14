#!/usr/bin/env python3
"""S2 — test de la logique de sélection quickstart (VRAM mockée, sans GPU).

Vérifie que recommend() choisit le plus gros modèle qui TIENT dans la VRAM, avec
le bon quant selon le matériel. Collectable par pytest (pas de GPU requis) ET
exécutable en script.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vramancer.quickstart as qs


def _mock(total_gb, blackwell):
    qs._detect = lambda: (total_gb, blackwell, [f"MOCK {total_gb}GB"])


CASES = [
    # (total VRAM, blackwell, use_case, attendu params_b, attendu quant)
    (39.0, True,  "code-assistant", 32, "nvfp4"),  # 2x notre matériel → 32B nvfp4
    (16.0, True,  "code-assistant", 14, "nvfp4"),  # 1 Blackwell 16GB → 14B nvfp4
    (24.0, False, "chat",           32, "nf4"),    # 3090 seule: 32B nf4 tient (22.6<24)
    (8.0,  False, "code-assistant",  7, "nf4"),    # 8GB → 7B nf4
    (4.0,  False, "chat",            3, "nf4"),    # 4GB → 3B nf4 (bf16 3B=8.7GB ne tient pas)
    (40.0, False, "summarize",      14, "bf16"),   # gros VRAM, usage plafonné 14B → bf16 (qualité)
]


def test_quickstart_selection():
    """Chaque cas : bon modèle, bon quant, et le besoin VRAM tient dans le budget."""
    for total, bw, uc, want_p, want_q in CASES:
        _mock(total, bw)
        r = qs.recommend(uc)
        assert r["params_b"] == want_p, f"{uc} {total}GB: {r['params_b']}B != {want_p}B attendu"
        assert r["quant"] == want_q, f"{uc} {total}GB: quant {r['quant']} != {want_q}"
        assert r["vram_need_gb"] <= total + 0.05, f"{uc} {total}GB: besoin {r['vram_need_gb']} > budget"


def test_tight_case():
    """VRAM minuscule → plus petit modèle + flag tight."""
    _mock(1.0, False)
    r = qs.recommend("chat")
    assert r.get("tight") is True


def _run_as_script():
    fails = 0
    for total, bw, uc, want_p, want_q in CASES:
        _mock(total, bw)
        r = qs.recommend(uc)
        ok = (r["params_b"] == want_p and r["quant"] == want_q and r["vram_need_gb"] <= total + 0.05)
        fails += 0 if ok else 1
        print(f"[{'OK ' if ok else 'FAIL'}] {uc:14s} {total:>5}GB bw={bw!s:5} → "
              f"{r['model'].split('/')[-1]:32s} {r['params_b']}B {r['quant']:5} need~{r['vram_need_gb']}GB")
    _mock(1.0, False)
    tight = qs.recommend("chat").get("tight") is True
    fails += 0 if tight else 1
    print(f"[{'OK ' if tight else 'FAIL'}] cas serré → tight={tight}")
    print(f"\n{'TOUS OK' if fails == 0 else str(fails) + ' ÉCHECS'}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run_as_script() else 0)
