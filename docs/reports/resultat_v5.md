# resultat_v5.md déplacé

Voir [/resultat_v5.md](../../resultat_v5.md) à la racine du repo pour le résultat à jour.
- `usb4_distributed_vram.py` déprécié proprement (P7)
- TODO ouvert `turbo_engine:202` documenté dans `TECHNICAL_DEBT.md` (P9)

**Nouvelles capacités :**
- Browser HF fonctionnel end-to-end (recherche + chargement via `POST /api/models/load`) (P12)
- KV cache offload DRAM 200 GB cap via `VRM_KV_OFFLOAD_ENGRAM=1` (P13)
- `VRM_TRANSFER_ASYNC=1` + `direct_vram_copy_async` Rust PyO3 (P4)

**Hygiène :**
- 25 bench_*.{json,log,txt} déplacés vers `benchmarks/results/` (P8)
- `.gitignore` durci contre les artifacts à la racine (P8)
- Version 1.5.0 → 1.6.0, CHANGELOG promu, TECHNICAL_DEBT V5 refresh (P11)
- Benchmarks reproductibles livrés (P6, P13)

**Skipped honnêtes :**
- `[SKIPPED@P6]` : Qwen2.5-14B OOM 2-GPU (mémoire insuffisante en session active)
- `[PARTIAL@P13.1]` : DeepSeek-V4-Flash 158B >> 40GB VRAM, proxy Qwen2.5-7B-Instruct

**Verdict global V5 : PARTIAL — toutes les phases exécutées ou documentées honnêtement**

Les phases OOM/hardware-bound (P6, P13.1) ont des sorties propres et reproductibles.
Aucune régression introduite. 4 nouveaux tests. Version 1.6.0 livrée.

**Reste à faire (V6 candidat) :**
- ~193 silents excepts restants hors hot paths (P5 partiel)
- `TURBO_KV_CUDAGRAPH` : Phase 2 turbo_engine (StaticKVCache + CUDA Graph capture)
- `TURBO_KV_HMM_OFFLOAD` : migrer `_DramDict` shim vers vrai `HierarchicalMemoryManager`
- `VRM_TRANSFER_OVERLAP=1` gains mesurables (benchmark à faire)
- Qwen2.5-14B 2-GPU : libérer VRAM avant bench (tuer vLLM workers)
- DeepSeek-V4-Flash GGUF Q4 (~80GB) : nécessite >40GB VRAM ou NVMe offload
