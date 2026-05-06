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
**[DÉBLOQUÉ — RAM 192 GB confirmée après redémarrage Proxmox]**
- RAM après redémarrage : **191 Gi total, 185 Gi disponible** — `free -h` du 2026-05-06
- Config finale : RTX 3090 (24 GB) + RTX 5070 Ti (16 GB) = 40 GB VRAM + 185 Gi RAM système
- Plan :
  - DeepSeek Q4 (GGUF Q4_K_M ou AWQ INT4) dual-GPU
  - KV cache Flash qui déborde des GPU → ReBAR → RAM système (engram / hierarchical_memory)
  - Quantization cible : **Q4** (GGUF Q4_K_M ou AWQ INT4)
  - Skip if OOM toujours valide si modèle trop lourd même en Q4
## [P14] — Validation finale
- **Tests post-reboot (2026-05-06) :** 0 failed, 1064 passed, 60 skipped
- Fix appliqué : `test_nvfp4_returns_nvfp4_on_blackwell` — le test mockait `sys.modules["torch"]` mais `_get_quantization_mode()` utilise `_torch` (variable module-level liée à l'import). Fix : patch `core.backends._torch` + `core.backends._HAS_TORCH` directement.
- Contexte : driver NVIDIA non chargé après reboot Proxmox (`modprobe nvidia` → module absent pour kernel 6.8.0-111-generic). Test est désormais correctement isolé (ne dépend plus du driver GPU).
## [SUMMARY]
