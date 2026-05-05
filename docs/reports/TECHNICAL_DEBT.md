# Technical Debt — VRAMancer

> Dernière mise à jour : 2026-05 (V4 plan execution)
> Maintenu manuellement à chaque PR qui ajoute ou résout un stub.

## Nouveaux stubs documentés en V4 (skip tests propres ajoutés)

| ID | Fichier | Test de skip | Statut |
|----|---------|--------------|--------|
| VTP_L3 | csrc/vtp_core.cpp L54-62 | `tests/test_vtp_l3_stub.py` | Documenté P6.1 |
| DMABUF_WRITE | csrc/dmabuf_bridge.c | header STUB explicite | Documenté P6.2 |
| NAT_HOLE_PUNCH | core/network/nat_traversal.py | `tests/test_nat_traversal_stub.py` | Documenté P6.3 |

## Stubs réels (code en place mais incomplet)

| ID | Fichier | Ligne(s) | Description | Effort | Priorité |
|----|---------|----------|-------------|--------|----------|
| VTP_L3 | csrc/vtp_core.cpp | 54-62 | Router L3+ retourne `src.clone()` au lieu d'un vrai transport RDMA/Network | Moyen | Basse — VTP routing ne sert qu'en cluster multi-node, et le path AITP/RDMA via `core/network/network_transport.py` est déjà fonctionnel séparément |
| DMABUF_WRITE | csrc/dmabuf_bridge.c | header | `vrm_dmabuf_transfer` étapes 4-5 : dst mmap write pas implémenté ; le caller Python doit faire la copie finale via torch pinned memory | Gros | Basse — les stratégies CUDA P2P (RTX→RTX) couvrent 99% des cas. DMA-BUF sert pour cross-vendor (NVIDIA↔AMD) qui est rare |
| NAT_HOLE_PUNCH | core/network/nat_traversal.py | punch_hole + relay | UDP hole punching et TURN relay non testés en WAN réel ; STUN RFC 5389 est réel. NAT type classification non implémentée (toujours "none"/"nat66") | Moyen | Basse — mode LAN/intranet n'en a pas besoin |
| WEBGPU_BACKEND | _deprecated/backends_webgpu.py | global | POC déprécié : nodes jamais peuplées automatiquement, "Speculative Decoding" = batching optimiste, "Holographic Parity" = XOR simple | — | Aucune — déprécié, déplacé dans `_deprecated/` |
| BATCH_INFERENCE | _deprecated/batch_inference.py | — | Déprécié. `generate_batch_fn` n'a jamais été câblé. Remplacé par `core/continuous_batcher.py` | — | Aucune — déprécié |
| CUDA_GRAPH_MULTI_GPU | core/cuda_graph_decode.py | global | CUDA Graph fonctionne single-device uniquement. NCCL/P2P ops ne peuvent pas être capturées dans un graph. | Gros | Basse — single-GPU decode déjà bénéficie du graph ; multi-GPU pipeline parallel n'en a pas besoin |
| TURBO_KV_CUDAGRAPH | core/turbo_engine.py | ~202 | Phase 2 non implémentée : StaticKVCache + CUDA Graph capture pour decode single-GPU. Phase 1 DynamicCache fonctionne (52+ tok/s). | Gros | Moyenne — gain perf single-GPU significatif si fait (CUDA Graph elimine overhead Python/CUDA launch) |
| TRITON_SAMPLING_TOPK | core/triton_sampling.py | top-k | **DESCRIPTION CORRIGÉE (V4 P3) :** `fused_sample` n'est appelé qu'en mode multi-GPU pipeline (Path 2 backends.py). Single GPU utilise HF `.generate()` nativement. Avec `top_k=0` (défaut), le kernel Triton full-vocab est bien actif — le "fallback PyTorch" n'est PAS utilisé en pratique. La fast_topk branch est sous-utilisée (top_k=0 par défaut), mais changer ce défaut modifie le comportement de sampling (filtrage top_k qu'il n'y avait pas avant). | Moyen | Basse désormais — Triton est actif, pas de fallback PyTorch |

## Limitations connues (non-bugs, by design)

| Limitation | Contournement |
|------------|---------------|
| BnB quantization multi-GPU = upstream bug accelerate 1.13.0 | VRAMancer force single-GPU si BnB |
| Continuous batcher backpressure max_waiting_queue=256 | Ajustable si besoin via code |
| `_apply_neuroplasticity_score()` non-déterministe | By design — utilise les poids synaptiques live du Connectome (Hebbian) |
| CUDA Graph decode single-device seulement | NCCL inter-device non capturable — limitation PyTorch fondamentale |
| aitp_receiver XDP requiert CAP_NET_ADMIN | Fallback UDP gracieux — pas un bug |
| TransferManager P2P topology cached forever | Pas de hotplug GPU supporté en cours de session |
| ~~`_get_method_for()` retourne `CPU_STAGED` alors que `send_activation()` utilise le bypass Rust P2P (172-190 Gbps)~~ ✅ **Résolu V5 P2** — `TransportMethod.RUST_P2P` ajouté, `_get_method_for()` retourne `"RUST_P2P"` quand `_gpu_pipelines` cache la paire (src,dst), et les deux return statements Rust dans `_execute_transfer` utilisent désormais `RUST_P2P`. | — |
| ~~**CONTINUOUS_BATCHER_GENERATE_BYPASS**~~ ✅ **Résolu V5 P1** — Auto-start du batcher au premier appel `generate()` quand `VRM_CONTINUOUS_BATCHING=1`. Fix également `_unbatch_kv_cache` pour DynamicCache transformers 5.x et tuples de longueur > 2. | — |

## Stubs résolus depuis l'audit 2026-03 (pour traçabilité)

- ✅ `software_cxl.cpp` → `csrc/file_offload.cpp` (nom honnête, header explicite)
- ✅ `supervision_api.NODES` → désormais dynamique (peuplé par heartbeat)
- ✅ `_get_method_for()` label → `RUST_P2P` ajouté (V5 P2) — label honnête pour le bypass Rust GpuPipeline
- ✅ `CONTINUOUS_BATCHER_GENERATE_BYPASS` → auto-start dans `generate()` + fix `_unbatch_kv_cache` DynamicCache (V5 P1)
- ✅ `vram_lending` → testé réel RTX 3090 + 5070 Ti (register/borrow/reclaim/re-borrow OK)
- ✅ `hierarchical_memory` eviction/spill → réel (Rust `cxl_direct_memory_dump`)
- ✅ `block_router.RemoteExecutor` docstring → corrigée (n'est PAS zero-copy — safetensors round-trip)
- ✅ `dashboard/launcher.launch_cli_dashboard()` → existe via alias `dashboard/__init__.py`
- ✅ `routes_ops.py` GPU detection → multi-vendor (CUDA/ROCm/MPS/XPU)
- ✅ `backends_ollama.py` aiohttp leak → supprimé, sync requests uniquement
- ✅ `backends_vllm.py` OOM retry no-op → fix (halve max_tokens uniquement)
- ✅ `stream_manager` async executor jamais join() → `shutdown(wait=True)` + thread.join()
- ✅ `batch_inference.py` fallback séquentiel → déplacé dans `_deprecated/`
- ✅ `rust_core.detect_best_transport()` → probe `libibverbs.so.1` via libloading (pas un stub)
- ✅ `cli/dashboard.py` endpoint `/api/pipeline/status` → existe dans `routes_ops.py`
- ✅ `continuous_batcher.py` GIL → `_batch_prepare_requests()` avec fallback cascade GIL-free
- ✅ VM Proxmox P2P bloqué IOMMU → ReBAR activé (2026-05), Strategy 1.5 fonctionnelle

## Comment mettre à jour ce fichier

Lors de chaque PR :
1. Si un stub est **résolu** → déplacer depuis "Stubs réels" vers "Stubs résolus" avec date.
2. Si un stub est **ajouté** → ajouter dans "Stubs réels" avec ID, effort, priorité.
3. Si une limitation change → mettre à jour "Limitations connues".
