# VRAMancer — Cœur produit

## Pitch

VRAMancer est un moteur d'inférence LLM pour **GPU hétérogènes**. Il découpe un modèle proportionnellement à la VRAM libre de chaque GPU, oriente l'inférence bloc par bloc, et gère les transferts inter-GPU avec plusieurs stratégies (P2P, CPU-staged, RDMA).

**Preuve concrète** : Qwen2.5-14B exécuté sur RTX 3090 (23.6 GB) + RTX 5070 Ti (15.5 GB) = **6.0 tok/s**, alors que ni l'un ni l'autre ne peut charger le modèle seul (OOM).

## Pipeline

```
HTTP API (FastAPI ou Flask)
   ↓
InferencePipeline (core/inference_pipeline.py)
   ↓
Backend (core/backends.py — HF/vLLM/llama.cpp/Ollama)
   ↓
ModelSplitter (core/model_splitter.py)  → plan de placement par couche
   ↓
Scheduler (core/scheduler.py)  → allocation des blocs
   ↓
TransferManager (core/transfer_manager.py)  → P2P / CPU-staged
   ↓
ComputeEngine (core/compute_engine.py)  → forward
```

## Composants clés

### `core/model_splitter.py`
Split VRAM-proportionnel basé sur la **mémoire libre** (pas totale). Pondération compute-aware. Si `LayerProfiler` est disponible, le split devient DP-optimal.

### `core/transfer_manager.py`
Stratégies de transfert tensor entre GPU : `cudaMemcpyPeerAsync`, torch P2P, PyO3/Rust `direct_vram_copy`, NCCL group, staging via mémoire pinned host. Détection topologie au démarrage.

### `core/scheduler.py`
Alloue les blocs aux GPU et gère le routage par couche.

### `core/paged_attention.py`
KV cache paginé style vLLM. Compatible compression KV via `kv_quantizer.py` (PolarQuant + QJL ≈ 3.5 bits/dim). Support GQA head-batching.

### `core/inference_pipeline.py`
Orchestrateur central. `load()` câble tous les sous-systèmes. `generate()` route vers : speculative decoding → continuous batcher (si actif) → forward direct.

## Variables d'environnement essentielles

| Variable | Rôle |
|---|---|
| `VRM_QUANTIZATION` | `nvfp4` (Blackwell), `nf4` (BnB), `int8` (BnB), vide = BF16 |
| `VRM_KV_COMPRESSION` | `turboquant` pour activer PolarQuant + QJL |
| `VRM_CONTINUOUS_BATCHING` | `1` active le batcher multi-requêtes |
| `VRM_PARALLEL_MODE` | `pp` (pipeline) ou `tp` (tensor parallel + NCCL) |
| `VRM_VRAM_LENDING` | `1` active le pool de VRAM cross-GPU |
| `VRM_TRUST_REMOTE_CODE` | `1` autorise `trust_remote_code` sur les modèles HF |

## Limites connues

- **VM Proxmox** : seule la stratégie 4 (CPU-staged pinned) fonctionne ; P2P bloqué par IOMMU.
- **BnB multi-GPU** : bug upstream `transformers 5.3.0 + accelerate 1.13.0` → forcé en single-GPU.
- **Continuous batcher** : limite `max_waiting_queue=256`, GIL CPython inhérent.
- **Rust transport** : `detect_best_transport()` retourne `ZeroCopyTcp` (stub).

Pour les détails complets, voir `.github/copilot-instructions.md`.
