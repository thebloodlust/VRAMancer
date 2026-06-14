#!/usr/bin/env python3
"""S2 — test de la logique de sélection quickstart (VRAM mockée, sans GPU).

Vérifie que recommend() choisit le plus gros modèle qui TIENT dans la VRAM, avec
le bon quant selon le matériel. La sélection est la partie qui décide tout.
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

failures = 0
for total, bw, uc, want_p, want_q in CASES:
    _mock(total, bw)
    r = qs.recommend(uc)
    ok = (r["params_b"] == want_p and r["quant"] == want_q
          and r["vram_need_gb"] <= total + 0.05)
    tag = "OK " if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"[{tag}] {uc:14s} {total:>5}GB bw={bw!s:5} → {r['model'].split('/')[-1]:32s} "
          f"{r['params_b']}B {r['quant']:5} need~{r['vram_need_gb']}GB "
          f"(attendu {want_p}B {want_q})")

# Cas serré : rien ne tient → plus petit modèle, flag tight
_mock(1.0, False)
r = qs.recommend("chat")
tight_ok = r.get("tight") is True
print(f"[{'OK ' if tight_ok else 'FAIL'}] VRAM 1GB → tight={r.get('tight')} modèle={r['model'].split('/')[-1]}")
if not tight_ok:
    failures += 1

print(f"\n{'TOUS OK' if failures == 0 else str(failures) + ' ÉCHECS'}")
sys.exit(1 if failures else 0)
