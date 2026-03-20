# VRAMancer — Architecture & Structure du Projet

> Orchestrateur multi-GPU pour l'inférence LLM distribuée  
> **~27 400 LOC** · **104 fichiers Python** · **5 fichiers C/C++/CUDA** · **1 module Rust**

---

## Vue d'ensemble

```
VRAMancer/
├── core/                    # Noyau métier (42 modules Python)
│   ├── api/                 # API Flask + routes ops (5 modules)
│   ├── network/             # Réseau, transport, mesh P2P (32 modules)
│   ├── orchestrator/        # Placement, rééquilibrage (4 modules)
│   └── security/            # Auth, remote access, checks (3 modules)
├── rust_core/               # Extension native Rust/Tokio (PyO3)
├── csrc/                    # Extensions C/C++/CUDA (XDP, VTP, CXL)
├── dashboard/               # CLI + Web dashboard (4 modules)
├── vramancer/               # CLI entrypoint + wrapper (5 modules)
├── tests/                   # Tests pytest (45 fichiers)
├── scripts/                 # Scripts utilitaires (8 fichiers)
├── docs/                    # Documentation (21 fichiers .md)
├── monitoring/              # Config Prometheus/Grafana/Alertmanager
├── examples/                # Exemples d'utilisation (4 scripts)
├── config/                  # Configs Docker, systemd, env profiles
└── build/                   # Web installer
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

## `core/` — Noyau métier (42 modules)

### Pipeline & Backends

| Fichier | Rôle | LOC |
|---------|------|-----|
| `inference_pipeline.py` | Chef d'orchestre : Backend→Scheduler→Monitor→Transfer→Stream→Compute | ~800 |
| `backends.py` | Factory LLM : `select_backend()` → HuggingFaceBackend + KVCacheBlock | ~900 |
| `backends_vllm.py` | vLLMBackend (extrait) | ~150 |
| `backends_ollama.py` | OllamaBackend (extrait) | ~200 |
| `backends_deepspeed.py` | DeepSpeedBackend *(code mort, jamais importé)* | ~300 |
| `backends_tensorrt.py` | TensorRTBackend *(code mort, jamais importé)* | ~250 |
| `backends_webgpu.py` | WebGPUBackend + NodeManager (experimental) | ~400 |
| `compute_engine.py` | Exécution réelle nn.Module (forward, torch.compile) | ~300 |
| `tokenizer.py` | Tokenisation HF + fallback regex | ~200 |

### Scheduling & Mémoire

| Fichier | Rôle | LOC |
|---------|------|-----|
| `scheduler.py` | SimpleScheduler : allocate/release/predict/migrate blocks | ~500 |
| `block_router.py` | Routage VRAM-aware GPU/CPU/NVMe/réseau | ~400 |
| `hierarchical_memory.py` | 6 niveaux (VRAM→DRAM→NVMe→réseau) + scoring LRU/LFU | ~500 |
| `holographic_memory.py` | Mémoire holographique + erasure coding XOR | ~300 |
| `paged_attention.py` | KV cache paginé (pages physiques/virtuelles) | ~400 |
| `memory_balancer.py` | Rééquilibrage mémoire inter-GPU | ~200 |
| `memory_block.py` | Structures MemoryBlock | ~100 |
| `memory_monitor.py` | Surveillance pression mémoire | ~200 |
| `stream_manager.py` | Prefetch, swap, eviction, monitoring background | ~500 |
| `continuous_batcher.py` | Batching continu *(TRONQUÉ, incomplet)* | ~250 |

### Transport GPU-to-GPU

| Fichier | Rôle | LOC |
|---------|------|-----|
| `transfer_manager.py` | P2P → NCCL → CPU-staged, cross-vendor bridge | ~800 |
| `transport_factory.py` | Factory : SAME_GPU/SAME_NODE/SAME_RACK/REMOTE | ~400 |
| `model_splitter.py` | Split VRAM-proportionnel (FREE memory) ou DP-optimal | ~400 |
| `cross_vendor_bridge.py` | AMD↔NVIDIA via DMA-BUF *(TRONQUÉ)* | ~200 |

### Monitoring & Métriques

| Fichier | Rôle | LOC |
|---------|------|-----|
| `monitor.py` | GPUMonitor : VRAM usage, detect_overload, ROCm fallback | ~500 |
| `metrics.py` | ~50 compteurs/gauges Prometheus (port 9108) | ~300 |
| `layer_profiler.py` | Profiling par couche (latence, FLOPS, mémoire) | ~400 |
| `benchmark.py` | Benchmark GPU compute + bandwidth | ~200 |
| `health.py` | Health checks composites | ~150 |

### Configuration & Utilitaires

| Fichier | Rôle | LOC |
|---------|------|-----|
| `config.py` | Config hiérarchique : defaults→YAML→env vars VRM_* | ~400 |
| `utils.py` | `detect_backend()` (cuda/rocm/mps/cpu), helpers | ~200 |
| `logger.py` | Logging structuré JSON, rotation | ~150 |
| `telemetry.py` | Export OpenTelemetry | ~300 |
| `tracing.py` | Tracing distribué | ~200 |
| `persistence.py` | Persistence SQLite | ~250 |
| `__init__.py` | Version (`1.5.0`), exports | ~50 |

### Modules avancés

| Fichier | Rôle | LOC |
|---------|------|-----|
| `compressor.py` | Compression zstd/lz4/gzip + quantization INT8/INT4 | ~400 |
| `speculative_decoding.py` | Décodage spéculatif *(TRONQUÉ)* | ~200 |
| `gpu_fault_tolerance.py` | Détection/récupération pannes GPU | ~300 |
| `gpu_interface.py` | Abstraction multi-accélérateur | ~200 |
| `hetero_config.py` | Configuration clusters hétérogènes | ~150 |
| `swarm_ledger.py` | Économie P2P : crédits, enchères, marketplace | ~400 |
| `vram_lending.py` | Prêt VRAM inter-nœuds *(TRONQUÉ)* | ~200 |
| `wake_on_inference.py` | Réveil de nœuds dormants | ~150 |
| `model_hub.py` | Registry modèles *(stub)* | ~100 |

---

## `core/api/` — API Flask (5 modules)

| Fichier | Rôle |
|---------|------|
| `registry.py` | PipelineRegistry (gestion modèles chargés) |
| `routes_ops.py` | Routes health/system/GPU (Blueprint extrait) |
| `validation.py` | Validation paramètres d'entrée |
| `batch_inference.py` | Inférence par lot |
| `circuit_breaker.py` | Pattern circuit breaker |

---

## `core/network/` — Réseau & Mesh P2P (32 modules)

### Protocole AITP (AI Transport Protocol)

| Fichier | Rôle |
|---------|------|
| `aitp_protocol.py` | Protocole UDP/IPv6 binaire (header 16 bytes) |
| `aitp_sensing.py` | Discovery multicast IPv6 (ff12::a1:b2:c3) |
| `aitp_fec.py` | Forward Error Correction pour AITP |
| `aitp_network_raid.py` | RAID réseau pour tenseurs |

### Transport réseau

| Fichier | Rôle |
|---------|------|
| `fibre_fastpath.py` | RDMA verbs → GPUDirect → ZeroCopy TCP (~700 LOC) |
| `llm_transport.py` | VTP : protocole binaire tensor-aware (~1000 LOC) |
| `transmission.py` | Sérialisation/envoi tenseurs |
| `transport.py` | Abstraction transport |
| `packet_builder.py` | Construction paquets binaires |
| `packets.py` | Structures de paquets |

### Cluster & Discovery

| Fichier | Rôle |
|---------|------|
| `cluster_discovery.py` | mDNS/ZeroConf + UDP broadcast + USB4 hot-plug |
| `cluster_master.py` | Nœud maître du cluster |
| `interface_selector.py` | Sélection interface réseau optimale |
| `vramancer_link.py` | Liens inter-nœuds |
| `trust_ring.py` | Anneau de confiance P2P |

### Exécution distribuée

| Fichier | Rôle |
|---------|------|
| `remote_executor.py` | Exécution distante de layers (**⚠️ pickle RCE**) |
| `swarm_inference.py` | Inférence distribuée swarm |
| `resource_aggregator.py` | Agrégation ressources cluster |

### Monitoring & Réparation

| Fichier | Rôle |
|---------|------|
| `network_monitor.py` | Monitoring réseau |
| `network_trace.py` | Tracing réseau |
| `supervision.py` | Supervision nœuds |
| `supervision_api.py` | API supervision (**⚠️ pas d'auth**) |
| `auto_repair.py` | Auto-réparation réseau |
| `actions.py` | Actions distantes (reboot, failover) (**⚠️ pas d'auth**) |

### Modules avancés

| Fichier | Rôle |
|---------|------|
| `connectome.py` | Cartographie topologique du cluster |
| `edge_iot.py` | Support IoT/Edge |
| `neural_compression.py` | Compression neurale *(stub)* |
| `speculative_stream.py` | Streaming spéculatif *(stub)* |
| `webgpu_node.py` | Nœuds WebGPU distants |
| `security.py` | Sécurité réseau |

---

## `core/orchestrator/` — Placement & Migration (4 modules)

| Fichier | Rôle |
|---------|------|
| `placement_engine.py` | Placement production : stratégies pluggables (profiled/vram/balanced) |
| `block_orchestrator.py` | Orchestration blocs : migration, rééquilibrage |
| `heterogeneous_manager.py` | Gestion clusters hétérogènes (GPU mixtes) |
| `adaptive_routing.py` | Routage adaptatif *(code mort)* |

---

## `core/security/` — Sécurité (3 modules)

| Fichier | Rôle |
|---------|------|
| `startup_checks.py` | Vérifications sécurité au démarrage |
| `remote_access.py` | Credentials remote admin (**⚠️ creds dans logs**) |

Le fichier `core/security.py` (racine) gère : token HMAC, rate limiting, RBAC, `install_security(app)`.

---

## `rust_core/` — Extension native Rust (PyO3 + Tokio)

```
rust_core/
├── Cargo.toml              # Dépendances : pyo3, tokio, hmac, sha2, cudarc
└── src/
    └── lib.rs              # 328 LOC — 11 fonctions exportées
```

**Fonctions clés :**
- `send_tensor_p2p()` / `receive_tensor_p2p()` — TCP async Tokio + HMAC-SHA256
- `direct_vram_load()` — Allocation CUDA directe via cudarc
- `generate_holographic_parity()` / `heal_holograph()` — Erasure coding XOR
- `sign_payload_fast()` / `verify_hmac_fast()` — Crypto 100× plus rapide que Python

---

## `csrc/` — Extensions C/C++/CUDA

| Fichier | Rôle | LOC |
|---------|------|-----|
| `aitp_xdp_bypass.c` | eBPF/XDP : interception paquets avant le kernel (PoC) | 87 |
| `vtp_core.cpp` | Routeur mémoire L1-L7 (PyBind11) | 47 |
| `vtp_cuda.cu` | `cudaMemcpyPeerAsync()` — P2P GPU direct | 22 |
| `software_cxl.cpp` | Émulation CXL : RAM↔NVMe zero-copy | 52 |
| `swarm_core.cpp` | Erasure coding XOR (auto-vectorisé AVX2) | 115 |

---

## `dashboard/` — Interface utilisateur

| Fichier | Rôle |
|---------|------|
| `cli_dashboard.py` | Dashboard terminal (curses/rich) |
| `dashboard_web.py` | Dashboard web Flask + WebSocket + Three.js 3D |
| `launcher.py` | Lanceur unifié CLI/Web |

---

## `vramancer/` — Point d'entrée CLI

| Fichier | Rôle |
|---------|------|
| `main.py` | Entrypoint principal (`python -m vramancer`) |
| `__main__.py` | Redirect vers main.py |
| `cli/dashboard_cli.py` | Sous-commande `vramancer dashboard` |
| `cli/swarm_cli.py` | Sous-commande `vramancer swarm` |
| `cli/telemetry_cli.py` | Sous-commande `vramancer telemetry` |

---

## `tests/` — 45 fichiers de test

Tests organisés par domaine : pipeline, API, backends, sécurité, transport, monitoring, etc.  
Conftest : `VRM_MINIMAL_TEST=1`, `VRM_API_TOKEN=testtoken`, mock torch si absent.  
Markers : `@slow`, `@integration`, `@smoke`, `@heavy`, `@network`.

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
| `VRM_LOG_JSON=1` | Logs structurés JSON |
| `VRM_TRACING=1` | Active OpenTelemetry |
| `VRM_SQLITE_PATH` | Persistence SQLite |
| `VRM_READ_ONLY=1` | Mode lecture seule |
