# Résultat Plan V5 (MASTER)

**Date début :** 2026-05-06
**Branche :** chore/sonnet-plan-v5
**Plan :** docs/reports/PLAN_ACTION_V5.md
**Base :** chore/sonnet-plan-v4 @ b8d412d

## [BASELINE]
- Date : 2026-05-06
- Branche source : chore/sonnet-plan-v4 @ b8d412d
- GPU0 : RTX 3090 24576 MiB | GPU1 : RTX 5070 Ti 16303 MiB
- torch : 2.11.0+cu130 | transformers : 5.8.0 | vLLM : 0.20.1
- Tests : **1 failed** (test_health_imports_fault_manager, pre-existing) | **1074 passed** | 42 skipped

## [P1] — ContinuousBatcher auto-start
**[NEGATIVE@P1.4]** — Auto-start dans `generate()` revert.

- Bench OFF (V4 baseline): N=1=27.95, N=4=14.32, N=8=8.95 tok/s total
- Bench ON (V5 avec auto-start): N=1=300s timeout, N=4=300s timeout
- Root cause: batcher decode loop incompatible avec transformers 5.x `DynamicCache` (exception silencieuse dans `_forward_single` → future ne se résout jamais → timeout 300s)
- Prefill fix gardé: `_unbatch_kv_cache` supporte maintenant `DynamicCache` (transformers 5.x) et tuples arbitraires (index au lieu d'unpack)
- Auto-start code revert, log mis à jour avec note `[NEGATIVE@P1.4]`
## [P2] — TransferManager labels honnêtes
- `TransportMethod.RUST_P2P` ajouté à l'enum
- `direct_vram_copy()` et `GpuPipeline.transfer()` retournent désormais `RUST_P2P`
- `_get_method_for()` retourne `"RUST_P2P"` si `_gpu_pipelines` cache la paire (src,dst)
- TECHNICAL_DEBT.md: entrée `_get_method_for() CPU_STAGED` marquée ✅ Résolu V5 P2

## [P3] — vLLM gap : use_cache + audit
- Single-GPU path: `gen_kwargs.setdefault("use_cache", True)` avant `model.generate()`
- Batch path: même fix après merge kwargs
- 67 tests backend passent
## [P4] — PyO3 transfer_async
## [P5] — Silent exceptions sweep
## [P6] — Hetero advantage bench
## [P7] — usb4_distributed_vram example
## [P8] — Repo root cleanup
## [P9] — TODO markers
## [P10] — Tests coverage
## [P11] — Documentation 1.6.0
## [P12] — HF browser load
## [P13] — DeepSeek + engram
**[BLOCKED — deep_gemm "Unsupported architecture" — hardware incompatibility H100 requis]**

**Tentatives :**
1. Run 1 : `RuntimeError: DeepGEMM backend is not available` → root cause : `libnvrtc.so.13` absent du LD path. Fix : `LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cu13/lib`. **RÉSOLU** → Run 2 lancé.
2. Run 2 : `RuntimeError: Assertion error (hyperconnection.hpp:56): Unsupported architecture` → root cause : hardware **définitif, non-contournable**.

**Diagnostic technique (run 2, 2026-05-06 16:02) :**
- Modèle chargé en 150.93s (46 shards, ~159 GB DRAM + 5.9 GB VRAM par GPU)
- Crash lors de `profile_run()` → `_dummy_run()` → `deepseek_v4.py:forward` → `mhc_pre()` → `tf32_hc_prenorm_gemm()` → `deep_gemm.hyperconnection.hpp:56`
- **Root cause** : DeepSeek-V4-Flash utilise MHC (Multi-Head Compression / HyperConnection), un mécanisme d'attention qui requiert le kernel `tf32_hc_prenorm_gemm` de deep_gemm
- Ce kernel utilise les instructions `wgmma` (warpgroup matrix multiply accumulate), spécifiques à l'ISA **Hopper (SM90)**
- `support_deep_gemm()` dans vLLM : `is_device_capability(90) or is_device_capability_family(100)`
  - RTX 3090 (SM 8.6) → **False**
  - RTX 5070 Ti (SM 12.0) → `is_device_capability(90)` True mais deep_gemm JIT assert fail sur SM 12.0 (famille consumer Blackwell inconnue de deep_gemm)
- **Aucun fallback** : `tf32_hc_prenorm_gemm` avec `_tf32_hc_prenorm_gemm_impl = None` → `_missing()` → RuntimeError. MHC est obligatoire dans l'architecture DeepSeek-V4.
- **Conclusion** : DeepSeek-V4-Flash nécessite H100 (SM90) ou B200 (SM100 datacenter). Consumer GPUs incompatibles.

**V5 STOP rule** : bloqué 2 fois sur deep_gemm (bloquer #1 libnvrtc.so.13 résolu, blocker #2 architecture matérielle incontournable) → STOP documenté.
## [P14] — Validation finale
- **Tests post-reboot (2026-05-06) :** 0 failed, 1064 passed, 60 skipped
- Fix appliqué : `test_nvfp4_returns_nvfp4_on_blackwell` — le test mockait `sys.modules["torch"]` mais `_get_quantization_mode()` utilise `_torch` (variable module-level liée à l'import). Fix : patch `core.backends._torch` + `core.backends._HAS_TORCH` directement.
- Contexte : driver NVIDIA non chargé après reboot Proxmox (`modprobe nvidia` → module absent pour kernel 6.8.0-111-generic). Test est désormais correctement isolé (ne dépend plus du driver GPU).
## [SUMMARY]
