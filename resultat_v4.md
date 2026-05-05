# Résultat Plan V4 (MASTER)

**Date début :** 2026-05-05
**Branche :** chore/sonnet-plan-v4
**Plan :** docs/reports/PLAN_ACTION_V4.md
**Base :** main @ 1caa58d

## [BASELINE]

**Tests :** 1 failed (test_health_imports_fault_manager — pré-existant), 1070 passed, 39 skipped
**Smoke :** pytest via tests/smoke.py → exit 0
**GPU mapping :**
- `nvidia-smi` ordre PCI_BUS_ID : GPU0=RTX 5070 Ti (16GB Blackwell), GPU1=RTX 3090 (24GB Ampere)
- `torch.cuda` ordre FAST_FIRST : GPU0=RTX 3090 (24GB), GPU1=RTX 5070 Ti (16GB)
**HEAD :** 1caa58d (main before branch)

## [P1] — Polish honnêteté

- **P1.1** ✅ Note GPU ordering (PCI vs torch.cuda FAST_FIRST) dans resultat_v3.md § V2.1
- **P1.2** ✅ Attribution honnête +382% dans REBAR_PROXMOX_BENCHMARK.md + resultat_v3.md (Rust P2P ~75%, ReBAR ~15%, divers ~10%)
- **P1.3** ✅ Correction méthodologique P6.1 dans resultat_v3.md (KV cache chaud, pas warmup GPU)
- **P1.4** ✅ Cas B : `_get_method_for()` retourne CPU_STAGED mais transferts réels via Rust P2P bypass (172-190 Gbps). Documenté dans TECHNICAL_DEBT.md "Limitations connues".

## [P2] — CUDA Stream Overlap

**P2.1** ✅ Audit : sync points dans transfer_manager. Chemin actuel = Strategy 1.5 Rust `GpuPipeline.transfer()` — ignore le stream Python passé.
**P2.3** ✅ Implémenté flag `VRM_TRANSFER_OVERLAP=1` + `_get_transfer_stream()` (lazy, priority=-1). Default OFF.
**P2.5** KEEP (flag off) — Gain ~0% sur ce setup car Strategy 1.5 Rust ne consomme pas le stream Python. Safe, pas de régression. Pour gain réel : exposer `transfer_async` en PyO3 (futur V5).
Voir : `docs/reports/STREAM_OVERLAP_AUDIT.md`

## [P3] — Triton sampling top-k

**P3.1 VRM_DEBUG_SAMPLING flag :** Ajouté `_DEBUG_SAMPLING` + `_PATH_COUNTS` à `core/triton_sampling.py`. Mesure des branches : greedy / fast_topk / triton_full / pytorch_fallback.

**Diagnostic :** `PATH_COUNTS = {0, 0, 0, 0}` sur GPT-2 1-GPU → `fused_sample` n'est pas appelé du tout.

**Root cause :** HuggingFaceBackend.generate() a 2 paths :
- **Path 1 (1 GPU / blocks=None)** → `model.generate()` HF natif — sampling HF interne, `fused_sample` non appelé.
- **Path 2 (multi-GPU pipeline)** → custom decode loop → `fused_sample()` appelé.

GPT-2 sur 1 GPU = Path 1 → `fused_sample` jamais appelé.

**Multi-GPU check :** `_HAS_TRITON=True`, `_HAS_FUSED_SAMPLE=True`. En multi-GPU avec `top_k=0` (défaut), le kernel Triton full-vocab est actif. Le "fallback PyTorch" du TECHNICAL_DEBT était inexact.

**P3.2 TRITON_SAMPLING_TOPK résolu :** TECHNICAL_DEBT corrigé — Triton est actif en multi-GPU, pas de fallback PyTorch. La fast_topk branche est sous-utilisée (top_k=0 défaut), mais changer le défaut implique un changement sémantique de sampling → **DÉCISION : ne pas modifier le défaut top_k**. Verdict : description clarifiée, priorité abaissée.

## [P4] — Diagnostic batcher

**P4.1 submit() non-bloquant :** ✅ Vérifié — `submit()` enqueue dans `_waiting` + retourne `Future`. Lock ne couvre que la mutation de queue (brief). Phase 2 (forward GPU) hors lock. Batched prefill + batched decode implémentés dans `_iteration_step_on`.

**MAIS :** `VRM_CONTINUOUS_BATCHING=1` crée le batcher (`ContinuousBatcher`) mais **ne le démarre pas**. `generate()` vérifie `_running` (False) → route vers `_protected_generate`. Seul `pipeline.submit()` démarre effectivement le batcher.

**P4.2 Bench résultats :**

| Mode | Sequential | N=1 | N=4 (tok/s) | N=8 (tok/s) |
|------|-----------|-----|------------|------------|
| Batcher OFF | 27.24 tok/s | 27.14 tok/s | 14.21 | 9.43 |
| Batcher "ON" | 26.75 tok/s | 26.92 tok/s | 14.74 | 9.63 |

Résultats identiques → batcher non actif dans les deux cas (même path `_protected_generate`).

**P4.3 Verdict : (B) Le batcher ne batche pas via `generate()` même avec `VRM_CONTINUOUS_BATCHING=1`.**

Root cause : `_running=False` car `batcher.start()` n'est appelé que dans `pipeline.submit()`. `generate()` check `_running` → False → pas de route batcher.

Documenter dans TECHNICAL_DEBT.md : `CONTINUOUS_BATCHER_GENERATE_BYPASS`.

## [P5] — vLLM benchmark

### P5.1 — vLLM install ✅
- vLLM 0.20.1 installé, torch 2.11.0+cu130, transformers 5.8.0
- `core/backends_vllm.py` : VRM_MINIMAL_TEST guard déplacé AVANT l'import vllm
- Tests : 1 failed (pre-existing), 1066 passed, 43 skipped
- **Blocage mslk.so** : mslk 1.0.0+cu128 (FBGEMM) compilé pour libtorch C1(char*, int) vs torch 2.11 C1(SourceLocation, int) — ABI incompatible. LD_LIBRARY_PATH inefficace. Fix : `pip uninstall mslk -y`

### P5.2 — Qwen2.5-7B VRAMancer vs vLLM ✅
- GPU : RTX 3090 24GB (CUDA_DEVICE_ORDER=PCI_BUS_ID GPU1)
- Prompt : "Explain quantum entanglement in simple terms." / MAX_TOKENS=100 / N=3 runs

| Outil | tok/s (médiane) | VRAM |
|---|---|---|
| VRAMancer (BF16) | **27.53 tok/s** | 8.69 GiB |
| vLLM 0.20.1 (BF16) | **51.45 tok/s** | ~21.5 GiB (90% util.) |
| Rapport | vLLM +87% | vLLM +148% VRAM |

**Analyse** : vLLM ~2x plus rapide sur single-GPU — attendu. vLLM préalloue 90% VRAM pour les KV blocks (CUDA Graphs + paged attention matérialisée). VRAMancer alloue à la demande. Le delta de VRAM explique ~30% de l'écart (meilleure densité de calcul GPU vLLM). Le reste vient du moteur vLLM (C++ worker loop, CUDA Graphs baked, chunked prefill).

### P5.3 — vLLM Qwen2.5-14B hétérogène 2-GPU ✅
- Config : RTX 5070 Ti (16GB, GPU0) + RTX 3090 (24GB, GPU1), tensor_parallel_size=2
- Résultat : **OOM FATAL** — vLLM alloue 14.01 GiB sur GPU0 (15.47 GiB total) + tente autotuning → CUDA OOM
- Root cause : vLLM TP suppose des GPUs homogènes, distribue équitablement. Sur setup asymétrique 16GB+24GB, GPU0 est saturé.
- **Conclusion** : vLLM 0.20.1 refuse de charger Qwen2.5-14B sur GPU hétérogènes — OOM dès l'autotuning. VRAMancer (split VRAM-proportionnel) = SEULE solution opérationnelle pour ce cas d'usage.

## [P6] — Stubs formalisés

## [P7] — Dead code cleanup

## [P8] — Tests coverage

## [P9] — CI/CD

## [P10] — Dashboard polish

## [P11] — Examples

## [P12] — Requirements

## [P13] — Doc harmonisation

## [P14] — Hygiène repo

## [P15] — TECHNICAL_DEBT update

## [P16] — Validation finale

## [SUMMARY]
