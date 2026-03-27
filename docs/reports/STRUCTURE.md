# VRAMancer — Architecture & Structure du Projet

> Orchestrateur multi-GPU hétérogène pour l'inférence LLM distribuée  
> **~33 300 LOC Python** · **83 modules .py dans core/** · **6 fichiers C/C++/CUDA** · **1 module Rust (1 077 LOC)**  
> **Version 1.5.0** — 26 mars 2026

---

## Vue d'ensemble

```
VRAMancer/
├── core/                    # Noyau métier (83 modules Python, ~33 300 LOC)
│   ├── api/                 # API Flask + routes ops (6 modules, ~1 090 LOC)
│   ├── network/             # Réseau, transport, mesh P2P (16 modules, ~6 730 LOC)
│   ├── orchestrator/        # Placement, rééquilibrage (4 modules, ~830 LOC)
│   └── security/            # Auth, startup checks (2 modules, ~560 LOC)
├── rust_core/               # Extension native Rust/Tokio (PyO3, 1 077 LOC)
├── csrc/                    # Extensions C/C++/CUDA (6 fichiers, ~750 LOC)
├── dashboard/               # CLI + Web dashboard (4 modules, ~400 LOC)
├── vramancer/               # CLI entrypoint + wrapper
├── tests/                   # Tests pytest (63 fichiers, ~12 100 LOC, 853 pass)
├── benchmarks/              # Benchmarks GPU (6 scripts)
├── scripts/                 # Scripts utilitaires
├── docs/                    # Documentation (~25 fichiers .md)
├── monitoring/              # Config Prometheus/Grafana/Alertmanager
├── examples/                # Exemples d'utilisation
├── config/                  # Configs Docker, systemd, env profiles
└── _deprecated/             # Modules archivés (DeepSpeed, TensorRT, etc.)
```

---

## Flux principal d'inférence

```
Requête HTTP (OpenAI-compatible)
       │
       ▼
┌─ production_api.py ─────────────────────────┐
│  Flask + HMAC/RBAC (security.py)            │
│  /v1/completions · /api/generate · /api/infer│
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─ inference_pipeline.py ─────────────────────┐
│  Chef d'orchestre central (singleton)       │
│  load() → generate() → infer()             │
└──┬──────────────┬───────────────┬───────────┘
   │              │               │
   ▼              ▼               ▼
backends.py   scheduler.py   transfer_manager.py
 (HF/vLLM/     (allocate/     (P2P/NCCL/
  Ollama)       release        CPU-staged)
                blocks)
   │              │               │
   ▼              ▼               ▼
model_splitter  monitor.py    transport_factory.py
 (VRAM-prop.)   (GPU poll)    (locality → transport)
```

---

## `core/` — Noyau métier (83 modules)

### Pipeline & Backends

| Fichier | Rôle | LOC |
|---------|------|-----|
| `inference_pipeline.py` | Chef d'orchestre : Backend→Scheduler→Monitor→Transfer→Stream→Compute→Lending | ~1 450 |
| `backends.py` | Factory LLM : `select_backend()` → HuggingFaceBackend + KVCacheBlock + split_model | ~1 910 |
| `backends_llamacpp.py` | LlamaCppBackend : GGUF via llama-cpp-python, multi-GPU tensor_split, HF Hub auto-download | ~396 |
| `backends_vllm.py` | vLLMBackend (extrait) | ~245 |
| `backends_ollama.py` | OllamaBackend REST bridge | ~184 |
| `backends_webgpu.py` | WebGPUBackend + NodeManager *(POC/template)* | ~263 |
| `compute_engine.py` | Exécution réelle nn.Module (forward multi-accelerateur) | ~256 |
| `tokenizer.py` | Tokenisation HF + fallback regex | ~39 |
| `continuous_batcher.py` | Batching continu : chunked prefill, token-level batching | ~909 |
| `speculative_decoding.py` | Décodage spéculatif : auto-mapping draft models par famille | ~287 |

### Quantization & Performance

| Fichier | Rôle | LOC |
|---------|------|-----|
| `nvfp4_direct.py` | DirectFP4 bypass : remplace NVFP4Tensor par plain buffers + `_scaled_mm` | ~343 |
| `triton_gemv_nvfp4.py` | Kernel Triton GEMV pour NVFP4 (nibble unpack, per-device LUT) | ~160 |
| `triton_gemv.py` | Kernels Triton GEMV generiques | ~267 |
| `triton_sampling.py` | Kernel Triton : fused temperature+softmax sampling | ~191 |
| `turboquant.py` | TurboQuant : Walsh-Hadamard → polar → QJL 1-bit (~3.5 bits/dim) | ~305 |
| `turbo_engine.py` | Engine TurboQuant integration | ~1 109 |
| `compressor.py` | Compression zstd/lz4/gzip + quantization INT8/INT4 | ~302 |

### Scheduling & Mémoire

| Fichier | Rôle | LOC |
|---------|------|-----|
| `scheduler.py` | SimpleScheduler : allocate/release/predict/migrate blocks | ~474 |
| `block_router.py` | Routage VRAM-aware GPU/CPU/NVMe/réseau | ~496 |
| `paged_attention.py` | KV cache paginé + TurboQuant compress-on-evict + VRAM lending | ~1 138 |
| `paged_attention_cuda.py` | Wrapper JIT CUDA kernel PagedAttention (warp-level softmax, GQA) | ~234 |
| `hierarchical_memory.py` | 6 niveaux (VRAM→DRAM→NVMe→réseau) + scoring LRU/LFU | ~991 |
| `parity_memory.py` | XOR parity memory (ex-holographic_memory) | ~214 |
| `vram_lending.py` | Pool de prêt VRAM inter-GPU : leases, scoring, reclaim auto | ~1 059 |
| `stream_manager.py` | Prefetch, swap, eviction, monitoring background | ~404 |
| `memory_balancer.py` | Rééquilibrage mémoire inter-GPU (LRU) | ~100 |
| `memory_block.py` | Structures MemoryBlock | ~28 |
| `memory_monitor.py` | Surveillance pression mémoire | ~89 |
| `block_metadata.py` | Metadata blocks | ~57 |

### Transport GPU-to-GPU

| Fichier | Rôle | LOC |
|---------|------|-----|
| `transfer_manager.py` | Chaîne de strategies : CrossVendor→P2P→Rust→ReBAR→Pipelined→NCCL→CPU-staged | ~930 |
| `cross_vendor_bridge.py` | Bridge AMD↔NVIDIA, ReBarTransport (full-window > 4 GB), PipelinedTransport | ~1 364 |
| `transport_factory.py` | Factory par localité : SAME_GPU/SAME_NODE/SAME_RACK/REMOTE | ~307 |
| `model_splitter.py` | Split VRAM-proportionnel (FREE memory) ou DP-optimal, support MPS | ~423 |
| `tensor_parallel.py` | TP column/row-parallel + NCCL all-reduce (GPT-2 + Llama) | ~532 |

### Monitoring & Métriques

| Fichier | Rôle | LOC |
|---------|------|-----|
| `monitor.py` | GPUMonitor : VRAM usage, detect_overload, ROCm fallback, polling thread | ~604 |
| `metrics.py` | ~35 compteurs/gauges Prometheus | ~167 |
| `layer_profiler.py` | Profiling par couche (latence, FLOPS, mémoire), DP placement | ~739 |
| `benchmark.py` | Benchmark GPU compute + bandwidth (4 modes) | ~511 |
| `health.py` | Health checks composites | ~327 |
| `gpu_fault_tolerance.py` | State machine HEALTHY→DEGRADED→FAILED→OFFLINE | ~769 |

### Configuration & Utilitaires

| Fichier | Rôle | LOC |
|---------|------|-----|
| `config.py` | Config hiérarchique : defaults→YAML→env vars VRM_*, multi-OS, hot-reload | ~312 |
| `utils.py` | `detect_backend()` (cuda/rocm/mps/cpu), GPU helpers, BasicTokenizer | ~443 |
| `logger.py` | Logging structuré JSON/Rich/File | ~78 |
| `tracing.py` | OpenTelemetry tracing | ~86 |
| `telemetry.py` | Export telemetry binaire *(inutilisé)* | ~89 |
| `persistence.py` | Persistence SQLite CRUD | ~64 |
| `auth_strong.py` | JWT + PBKDF2-HMAC SHA256 | ~129 |
| `__init__.py` | Version (`1.5.0`), factory stubs test | ~57 |

### Modules réseau & cluster

| Fichier | Rôle | LOC |
|---------|------|-----|
| `hetero_config.py` | Auto-detection GPU hétérogène, VM, P2P, tiers, lending policy | ~595 |
| `gpu_interface.py` | Abstraction multi-accélérateur | ~75 |
| `swarm_ledger.py` | Ledger SQLite crédits P2P *(déconnecté)* | ~271 |
| `wake_on_inference.py` | Réveil de nœuds dormants (WoL magic packets) | ~119 |
| `model_hub.py` | Registry modèles *(stub)* | ~69 |

---

## `core/api/` — API Flask (6 modules, ~1 090 LOC)

| Fichier | Rôle | LOC |
|---------|------|-----|
| `batch_inference.py` | Inférence par lot *(fallback séquentiel — generate_batch_fn jamais fourni)* | ~324 |
| `routes_ops.py` | Routes health/system/GPU (Blueprint extrait) | ~310 |
| `registry.py` | PipelineRegistry (gestion modèles chargés) | ~185 |
| `circuit_breaker.py` | Pattern circuit breaker (protège timeout + SSE) | ~179 |
| `validation.py` | Validation paramètres d'entrée + prompt length | ~74 |
| `__init__.py` | Exports | ~19 |

---

## `core/network/` — Réseau & Mesh P2P (16 modules, ~6 730 LOC)

### Transport réseau

| Fichier | Rôle | LOC |
|---------|------|-----|
| `fibre_fastpath.py` | RDMA verbs (ibverbs/RoCE) → GPUDirect RDMA → ZeroCopy TCP | ~902 |
| `llm_transport.py` | VTP : protocole binaire tensor-aware *(incomplet, 60%)* | ~1 471 |
| `transmission.py` | Sérialisation/envoi tenseurs | ~175 |
| `transport.py` | Abstraction transport | ~156 |
| `packets.py` | Structures de paquets *(remplacé par transmission.py)* | ~35 |

### Protocole AITP (AI Transport Protocol)

| Fichier | Rôle | LOC |
|---------|------|-----|
| `aitp_protocol.py` | Protocole UDP/IPv6 binaire (header 16 bytes) | ~263 |
| `aitp_sensing.py` | Discovery multicast IPv6 | ~231 |
| `aitp_fec.py` | Forward Error Correction GF(2^8) Cauchy Reed-Solomon | ~248 |
| `aitp_receiver.py` | Réception AITP | ~389 |

### Cluster & Discovery

| Fichier | Rôle | LOC |
|---------|------|-----|
| `cluster_discovery.py` | mDNS + UDP broadcast + Bully leader election | ~925 |
| `nat_traversal.py` | STUN RFC 5389, UDP hole punch, relay, ULA IPv6 | ~372 |
| `connectome.py` | Cartographie Hebbian (synapse weights, EMA) | ~199 |
| `edge_api.py` | Flask blueprint edge device lifecycle | ~191 |
| `supervision_api.py` | API supervision + HA delta sync zstd/lz4/zlib | ~546 |
| `webgpu_node.py` | Nœuds WebGPU distants *(orphelin)* | ~507 |

### Modules cassés/stubs

| Fichier | Rôle | LOC |
|---------|------|-----|
| `interface_selector.py` | Sélection interface réseau *(import cassé)* | ~61 |
| `security.py` | Sécurité réseau *(logging invalide)* | ~47 |
| `vramancer_link.py` | Re-export 2 lignes *(inutile)* | ~8 |

---

## `core/orchestrator/` — Placement & Migration (4 modules, ~830 LOC)

| Fichier | Rôle | LOC |
|---------|------|-----|
| `placement_engine.py` | Placement production : stratégies pluggables (profiled/vram/balanced), DP-optimal | ~348 |
| `heterogeneous_manager.py` | Gestion clusters hétérogènes (GPU mixtes, scoring) | ~312 |
| `block_orchestrator.py` | Orchestration blocs : migration, rééquilibrage | ~167 |
| `__init__.py` | Redirect vers block_orchestrator | ~4 |

---

## `core/security/` — Sécurité (2 modules, ~560 LOC)

| Fichier | Rôle | LOC |
|---------|------|-----|
| `__init__.py` | Token + HMAC + RBAC + rate limiting + CSP, `install_security(app)` | ~479 |
| `startup_checks.py` | Vérifications sécurité au démarrage | ~83 |

> Note : `remote_access.py` déplacé vers `_deprecated/`. L'ancien `zero_trust.py` était un shim redirigé vers `startup_checks.py`.

Le fichier `core/production_api.py` (~1 146 LOC) implémente l'API Flask OpenAI-compatible avec `create_app()`, les endpoints `/v1/completions`, `/api/generate`, `/api/infer`, `/api/models/load`.

---

## `rust_core/` — Extension native Rust (PyO3 + Tokio, 1 077 LOC)

```
rust_core/
├── Cargo.toml              # Dépendances : pyo3, tokio, hmac, sha2, libloading
└── src/
    └── lib.rs              # 1 077 LOC — transport async + CUDA FFI
```

**Fonctions clés :**
- `GpuPipeline` — Pipeline persistant avec streams/events/pinned buffers pré-alloués
- `async_gpu_transfer()` — DtoH/HtoD overlappé sur CUDA streams séparés
- `send_tensor_chunked()` / `receive_tensor_chunked()` — Tokio TCP avec HMAC par chunk
- `generate_holographic_parity()` / `heal_holograph()` — Erasure coding XOR
- `sign_payload_fast()` / `verify_hmac_fast()` — Crypto HMAC-SHA256 via Rust

---

## `csrc/` — Extensions C/C++/CUDA (6 fichiers, ~750 LOC)

| Fichier | Rôle | LOC |
|---------|------|-----|
| `paged_attention_kernel.cu` | Kernel CUDA PagedAttention decode : warp-level online softmax, GQA, fp16/fp32. **8.8x@ctx64** | 361 |
| `aitp_xdp_bypass.c` | eBPF/XDP : interception paquets avant le kernel (PoC) | 145 |
| `swarm_core.cpp` | Erasure coding XOR (auto-vectorisé AVX2) | 98 |
| `vtp_core.cpp` | Routeur mémoire L1-L7 (PyBind11) | 67 |
| `software_cxl.cpp` | Émulation CXL : RAM↔NVMe *(stub)* | 46 |
| `vtp_cuda.cu` | `cudaMemcpyPeerAsync()` — P2P GPU direct | 34 |

---

## `dashboard/` — Interface utilisateur (~400 LOC)

| Fichier | Rôle |
|---------|------|
| `cli_dashboard.py` | Dashboard terminal *(appelle /api/gpu et /api/status qui n'existent pas)* |
| `dashboard_web.py` | Dashboard web Flask *(données GPU hardcodées)* |
| `launcher.py` | Lanceur unifié CLI/Web |

---

## `tests/` — 63 fichiers de test (~12 100 LOC, 853 pass, 38 skip)

Tests organisés par domaine : pipeline, API, backends, sécurité, transport, monitoring, quantization, lending, ReBAR.  
Conftest : `VRM_MINIMAL_TEST=1`, `VRM_API_TOKEN=testtoken`, mock torch si absent.  
Markers : `@slow`, `@integration`, `@smoke`, `@heavy`, `@network`.

---

## `_deprecated/` — Modules archivés

| Fichier | Raison |
|---------|--------|
| `backends_deepspeed.py` | DeepSpeed jamais intégré |
| `backends_tensorrt.py` | TensorRT jamais intégré |
| `adaptive_routing.py` | Code mort |
| `network_archive/` | Anciens modules réseau |

---

## `monitoring/` — Observabilité

| Fichier | Rôle |
|---------|------|
| `prometheus.yml` | Configuration scraping métriques |
| `alerting_rules.yml` | Règles d'alerte (VRAM plein, latence, erreurs) |
| `alertmanager.yml` | Configuration alertes (Slack, email) |
| `grafana_dashboard.json` | Dashboard Grafana pré-configuré |

---

## Variables d'environnement clés

| Variable | Effet |
|----------|-------|
| `VRM_MINIMAL_TEST=1` | Mode test : stubs partout (pas de torch) |
| `VRM_BACKEND_ALLOW_STUB=1` | Stub si backend LLM absent |
| `VRM_DISABLE_RATE_LIMIT=1` | Désactive rate limiting (CI) |
| `VRM_API_TOKEN` | Token auth API (défaut test : `testtoken`) |
| `VRM_PRODUCTION=1` | Mode production (validation stricte) |
| `VRM_QUANTIZATION` | `nvfp4` / `nf4` / `int8` / vide=BF16 |
| `VRM_KV_COMPRESSION` | `turboquant` (PolarQuant + QJL, ~3.5 bits/dim) |
| `VRM_KV_COMPRESSION_BITS` | Bits par angle polaire (défaut 3) |
| `VRM_VRAM_LENDING` | Active/désactive le VRAM Lending Pool (défaut `1`) |
| `VRM_LEND_RATIO` | Ratio max de VRAM libre prêtable (défaut `0.70`) |
| `VRM_RECLAIM_THRESHOLD` | Seuil d'utilisation déclenchant le reclaim (défaut `0.80`) |
| `VRM_CONTINUOUS_BATCHING=1` | Active le continuous batcher |
| `VRM_TRANSFER_P2P` | Force-désactive P2P (`0`/`false`) — utile en VM IOMMU |
| `VRM_LOG_JSON=1` | Logs structurés JSON |
| `VRM_TRACING=1` | Active OpenTelemetry |
| `VRM_SQLITE_PATH` | Persistence SQLite |
| `VRM_READ_ONLY=1` | Mode lecture seule |
