<div align="center">
  <h1>🚀 VRAMancer</h1>
  <p><b>The Heterogeneous AI Swarm / L'Essaim IA Asynchrone Zero-Trust</b></p>
  <p>
    <img src="https://img.shields.io/badge/Status-Beta-yellow" alt="Status">
    <img src="https://img.shields.io/badge/Hardware-NVIDIA%20%7C%20AMD%20%7C%20Apple-blue" alt="Hardware">
    <img src="https://img.shields.io/badge/Network-Zero--Trust%20%7C%20WebGPU-purple" alt="Network">
    <img src="https://img.shields.io/badge/Performance-Zero--Copy%20%7C%20Async-success" alt="Performance">
  </p>
</div>

---

### 🔥 Fonctionnalités Cœurs (Architecture Hautes-Performances) :

- **Inférence Asynchrone Native (GIL Bypassed) :** Intégration pur asynchrone des moteurs via `AsyncLLMEngine` (vLLM) et pointeurs CUDA directs (`data_ptr` via `execute_async_v2` de TensorRT) pour éviter toute latence CPU.
- **Topologie Réseau Zero-Trust (HMAC) :** Transferts de tenseurs P2P ultra-sécurisés, avec payloads scellés par **HMAC-SHA256**. Vous pouvez déployer VRAMancer dans un cloud distribué public sans risque d’exécution de code malicieux !
- **Mémoire Hiérarchique à 6 Niveaux :** Transfert fluide et intelligent de la VRAM (L1), vers NVLink/P2P (L2), vers la Pinned Memory classique/DRAM (L3), jusqu'au Swap NVMe ultra-rapide (L4) si nécessaire.
- **Hub Discovery & CLI "Docker-Like" :** Diagnostic local avancé des poids hébergés par l'API de HuggingFace. Localise les modèles quantifiés de nouvelle génération : **NVFP4 (Nvidia Blackwell), AWQ, GGUF, GPTQ**.

*(Pour les détails profonds du moteur interne asynchrone VRAMancer, lisez notre nouveau [📝 docs/architecture.md](./docs/architecture.md))*


## 🟢 Installation Simplifiée (Nouveau !)

VRAMancer supporte nativement un très grand nombre de matériels (NVIDIA, AMD ROCm, Apple MPS, **Intel XPU**, **Huawei NPU**, et WebGPU).

Pour vous simplifier la vie, deux méthodes recommandées :

### 1️⃣ Version Portable (Windows & Desktop)
Pas de Python ni de lignes de commandes nécessaires :
- **[Télécharger le fichier .ZIP dans l'onglet "Releases" de GitHub]**
- Décompressez et lancez `vramancer.exe` ! (Généré automatiquement)

### 2️⃣ Version Serveur (Linux & Datacenter via Docker)
Déploiement en 1 ligne sur RunPod, AWS, ou un serveur privé, avec drivers GPU inclus :
```bash
docker run --gpus all -p 8080:8080 ghcr.io/thebloodlust/vramancer:latest
```

## 🌍 What is VRAMancer? / Qu'est-ce que VRAMancer ?

**[EN]** VRAMancer is an enterprise-grade, heterogeneous multi-GPU inference engine. It allows you to pool VRAM from completely different devices (e.g., an EPYC server with RTX 3090, a Windows laptop with RTX 4060, and a Mac Mini M4) over a local network (Wi-Fi, Ethernet, USB4) to run massive LLMs like Llama 3 70B that wouldn't fit on a single machine.

**[FR]** VRAMancer est un moteur d'inférence multi-GPU hétérogène de niveau entreprise. Il vous permet de fusionner la VRAM d'appareils totalement différents (ex: un serveur EPYC avec RTX 3090, un PC portable Windows avec RTX 4060, et un Mac Mini M4) via le réseau local (Wi-Fi, Ethernet, USB4) pour faire tourner des modèles géants comme Llama 3 70B qui ne tiendraient pas sur une seule machine.

---

## ✨ Key Features / Fonctionnalités Clés

*   **🧠 Heterogeneous Pooling (CUDA + ROCm + MPS)**: Mix NVIDIA, AMD, and Apple Silicon seamlessly. / *Mélangez NVIDIA, AMD et Apple Silicon de manière transparente.*
*   **⚡ CPU-Staged PCIe Fallback (ReBar)**: Advanced automated fallback bridging asymmetric consumer GPUs (e.g., RTX 3090 + RTX 5070 Ti) when internal NVIDIA P2P/NVLink is locked. / *Pont automatique de secours par le CPU via le bus PCIe quand le transfert de VRAM à VRAM est physiquement bloqué par les drivers.*
*   **🌐 Swarm Inference (P2P)**: No master node required. Devices discover each other via mDNS and share the workload. / *Découverte automatique via mDNS, les appareils se partagent le calcul.*
*   **🕸️ Planetary WebGPU Offloading** *(Experimental 2026)*: Let any user's web browser join your cluster over the internet to lend its native GPU power via WebRTC & Speculative Network Decoding. / *Laissez n'importe quel navigateur web prêter sa carte graphique à travers le monde.*
*   **🛡️ Security**: API security with HMAC tokens, JWT, RBAC, and rate limiting. / *Sécurité API avec tokens HMAC, JWT, RBAC et rate limiting.*
*   **📊 Advanced Telemetry**: Built-in Prometheus metrics and Grafana dashboards. / *Métriques Prometheus et tableaux de bord Grafana intégrés.*

### Feature Maturity / Maturité des fonctionnalités

| Feature | Status | Notes |
|---|---|---|
| Multi-GPU model splitting (single node) | **Stable** | VRAM-proportional and profiler-based placement |
| Pipeline-parallel inference | **Stable** | GPU-to-GPU via CUDA P2P or CPU-staged fallback |
| OpenAI-compatible API | **Stable** | `/v1/completions`, `/v1/chat/completions`, SSE streaming |
| Continuous batching | **Stable** | Iteration-level scheduling with queue management |
| Paged KV cache | **Stable** | Prefix caching, copy-on-write, overflow to VRAM lending |
| VRAM lending pool | **Stable** | Cross-GPU cooperative memory pooling with lease tracking |
| 6-tier hierarchical memory | **Stable** | VRAM → DRAM → NVMe → Network → Swarm → WebGPU |
| Monitoring (Prometheus/Grafana) | **Stable** | 47 metrics, 24 panels, 16 alerting rules |
| Multi-node clustering (mDNS) | **Stable** | Auto-discovery and robust Zero-Trust network validation |
| RDMA / GPUDirect transport | **Stable** | Fully optimized via Zero-Copy; Fallback to TCP if RoCE/hardware unavailable |
| WebGPU offloading | **Stable** | Production-validated execution with JWT Auth |
| USB4 hot-plug | **Stable** | Native support on Linux (pyudev) & macOS (IOKit) |

---

## 🚀 Quickstart / Démarrage Rapide (Plug & Play)

**[EN] No Python required!** Download the standalone executable for your OS from the Releases page.
**[FR] Pas besoin de Python !** Téléchargez l'exécutable autonome pour votre OS depuis la page Releases.

### 1. Start the Main Node (Le Serveur)
```bash
# Linux (EPYC Server)
./vramancer-linux start --model "meta-llama/Llama-3-70b-instruct" --master
```

### 2. Connect your Devices (Les Renforts)
```bash
# Windows Laptop (RTX 4060)
vramancer.exe join --auto-discover

# Mac Mini (M4)
./vramancer-macos join --auto-discover
```

### 3. Chat! (Discutez !)
Open your browser and go to / *Ouvrez votre navigateur sur* : **http://localhost:5000**

---

## 🏗️ Architecture (V2)

VRAMancer uses a 6-tier hierarchical memory system and advanced network protocols:
*   **Tier 1**: VRAM (CUDA/ROCm/MPS)
*   **Tier 2**: System RAM (Pinned Memory)
*   **Tier 3**: NVMe SSD (PCIe Gen4/Gen5 Offloading)
*   **Tier 4**: FastPath Network (RDMA / Zero-copy TCP)
*   **Tier 5**: Swarm Peers (mDNS discovered nodes)
*   **Tier 6**: WebGPU Clients (Browsers via WebSockets)

### 🔮 Coming Soon (V2 Roadmap)
*   **Rust Core Rewrite (PyO3)**: Transition of the C++ *Software CXL* and Swarm modules to Rust. Eliminating theoretical Segfaults at the Python/Native boundary via the Rust Borrow Checker while keeping identical multi-threading AVX performance.
*   **Neural Compression**: On-the-fly INT4 quantization to kill network bandwidth bottlenecks.
*   **Wake-on-Inference**: Automatically wake up sleeping laptops (WoL) only when a massive prompt requires their VRAM.
*   **Live USB**: Boot any PC from a USB stick to instantly join the Swarm without touching the local hard drive.

---
*Built with ❤️ for the Open-Source AI Community.*
python install.py            # Linux / macOS / WSL
# ou: install.bat            # Windows
# ou: bash Install.sh        # Alternative bash

# Start the API server
python -m vramancer.main --api

# Load a model (auto-splits across available GPUs)
curl -X POST http://localhost:5000/api/models/load \
  -H "Content-Type: application/json" \
  -H "X-API-TOKEN: $VRM_API_TOKEN" \
  -d '{"model": "meta-llama/Llama-3.1-8B", "num_gpus": 2}'

# Generate text (OpenAI-compatible)
curl http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-TOKEN: $VRM_API_TOKEN" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 100}'
```

### Docker (recommended for production)

```bash
export VRM_API_TOKEN=your-secret-token
docker compose up -d

# API:          http://localhost:5030
# Grafana:      http://localhost:3000 (admin / vramancer)
# Prometheus:   http://localhost:9090
# Alertmanager: http://localhost:9093
```

### Run Tests

```bash
# Full test suite — no GPU required
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest tests/ -q --no-cov

# Real GPU tests (requires torch + GPU)
VRM_MINIMAL_TEST= pytest tests/test_real_gpu.py -v
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Flask API (OpenAI-compatible: /v1/completions, /v1/chat/...)   │
│  Security: Token + HMAC + RBAC + Rate Limiting + CORS           │
└───────────┬──────────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────────┐
│  InferencePipeline (orchestrator)                                │
│  ┌────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │  Backend    │  │ ContinuousBatcher│  │  VRAMLendingPool     │ │
│  │ (HF/vLLM)  │  │ iteration-level  │  │ cooperative cross-GPU│ │
│  │ split_model │  │ scheduling       │  │ VRAM pooling         │ │
│  └─────┬──────┘  └────────┬─────────┘  └──────────┬───────────┘ │
│  ┌─────▼──────┐  ┌────────▼─────────┐  ┌──────────▼───────────┐ │
│  │ ModelSplit  │  │ PagedKVCache     │  │  GPUBudget per-GPU   │ │
│  │ VRAM-prop.  │  │ multi-GPU +      │  │  lease tracking      │ │
│  │ or profiled │  │ overflow lending  │  │  graceful reclaim    │ │
│  └────────────┘  └──────────────────┘  └──────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  TransferManager: CUDA P2P > CPU-staged > NCCL (distrib)  │  │
│  │  VTP Protocol: GPUDirect RDMA > Zero-copy TCP > mmap      │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────────────────┐ │
│  │ GPUMonitor   │  │ HotPlugMonitor│  │ ClusterDiscovery      │ │
│  │ VRAM polling │  │ add/remove GPU│  │ mDNS + UDP broadcast  │ │
│  │ Prometheus   │  │ rebalancing   │  │ USB4 hot-plug         │ │
│  └─────────────┘  └───────────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### How Model Splitting Works

1. `model_splitter.detect_layers(model)` extracts transformer blocks
2. `model_splitter.split_model(blocks, gpu_list)` assigns layers proportionally to **free VRAM per GPU** (or uses `LayerProfiler` DP-optimal placement for heterogeneous compute)
3. During inference, `Backend.infer()` traverses blocks in order — when execution crosses a GPU boundary, `TransferManager.send_activation()` moves the hidden state via CUDA P2P (or CPU-staged fallback)
4. `ContinuousBatcher` runs all active requests through this pipeline in a single iteration step

### How VRAM Lending Works

When GPU 0 (3090, 24GB) runs out of KV cache space:

1. `PagedKVCache._borrow_overflow_page()` calls `VRAMLendingPool.borrow()`
2. The pool scores candidate lenders by: free capacity (40%) + PCIe speed (30%) + idle time (30%)
3. GPU 1 (5070 Ti, 16GB, with 4GB idle) creates a **lease** — a tracked region of pre-allocated VRAM
4. GPU 0 uses the borrowed VRAM for overflow KV pages
5. When GPU 1 needs its VRAM back: `reclaim(urgency=HIGH)` → graceful migration to CPU pinned memory → release
6. Priority ordering ensures low-priority leases are reclaimed first

### How Continuous Batching Works

Each iteration:
1. New requests enter the batch (up to `max_batch_size`)
2. Prefill requests run individually (different prompt lengths)
3. All decode-phase requests are coalesced into **one padded forward pass**
4. Completed requests are evicted, their paged KV slots freed
5. New requests can join at the next iteration

---

## Monitoring & Supervision

VRAMancer ships with full production monitoring out of the box.

### Metrics (47 Prometheus metrics)

| Family | Metrics | Examples |
|---|---|---|
| Inference | 3 | `infer_total`, `infer_errors_total`, `infer_latency_seconds` |
| GPU | 4 | `gpu_memory_used_bytes`, `gpu_transfer_ops_total`, `gpu_transfer_bandwidth_gbps`, `device_info` |
| Memory Tiering | 3 | `memory_promotions_total`, `memory_demotions_total`, `memory_evictions_total` |
| Network/FastPath | 4 | `fastpath_bytes_total`, `fastpath_latency_seconds`, `fastpath_interface_latency_seconds` |
| VTP Protocol | 5 | `vtp_tensors_sent_total`, `vtp_bytes_sent_total`, `vtp_kv_cache_ops_total`, `vtp_gpudirect_ops_total` |
| VRAM Lending | 5 | `lending_borrows_total`, `lending_reclaims_total`, `lending_active_leases`, `lending_bytes_lent_total`, `lending_pool_capacity_gb` |
| Batcher/KV | 6 | `batcher_batch_size`, `batcher_queue_depth`, `batcher_throughput_tok_s`, `paged_kv_used_pages`, `paged_kv_free_pages`, `paged_kv_borrowed_pages` |
| Tasks | 7 | `tasks_submitted_total`, `tasks_running`, `task_duration_seconds`, `task_duration_percentile` |
| API | 1 | `api_latency_seconds` (path, method, status) |
| Orchestrator | 4 | `orch_placements_total`, `orch_migrations_total`, `orch_rebalance_total`, `orch_hierarchy_moves_total` |
| HA | 2 | `ha_journal_rotations_total`, `ha_journal_size_bytes` |
| Telemetry | 1 | `telemetry_packets_total` |

### Grafana Dashboard

**24 panels** organized in 6 sections:

- 🔥 **Inference Overview** — RPS throughput, p50/p95/p99 latency, error rate, active tasks
- 🖥️ **GPU Memory & Utilization** — per-GPU memory gauge, transfer bandwidth, tasks per resource
- 📊 **Memory Tiering** — promotions/demotions/evictions rates, block hotness scores, hierarchy migrations
- 🌐 **Network & FastPath** — transfer rate (bytes/s), FastPath latency p95
- 🛡️ **API & Health** — endpoint latency distribution, task lifecycle (submitted/completed/failed)
- ⚙️ **Orchestrator & HA** — placement counts, migration/rebalancing rates, HA journal size

### Alerting (16 rules, 6 groups)

| Group | Alerts | Example |
|---|---|---|
| GPU | 3 | `GPUMemoryHigh` (>90%), `GPUMemoryCritical` (>10GiB), `NoGPUMetrics` |
| Inference | 4 | `HighInferenceErrorRate` (>5%), `HighInferenceLatency` (p95>10s), `InferenceStopped` |
| Tasks | 2 | `TaskQueueBacklog` (>16 running), `HighTaskFailureRate` (>10%) |
| API | 2 | `APILatencyHigh` (p95>5s), `APIDown` (up==0) |
| Transfers | 2 | `FastPathLatencyHigh` (p95>100ms), `GPUTransferBandwidthLow` (<1Gbps) |
| HA | 3 | `HAJournalLarge` (>100MiB), `FrequentRebalancing`, `HighMemoryEvictions` |

### Dashboard (CLI + Web)

- **CLI Dashboard**: Real-time ASCII display of GPU memory, system health, cluster status
- **Web Dashboard**: Flask + Socket.IO, dark theme, real-time memory hierarchy visualization, interactive block promote/demote, task management, p95/p99 metrics

### Docker Compose Stack

```bash
docker compose up -d
# inference (API + metrics) → :5030, :9108
# prometheus                → :9090
# grafana                   → :3000
# alertmanager              → :9093
```

---

## Multi-Node Cluster

```bash
# Machine A (master)
python -m vramancer.main --api --cluster-master

# Machine B (worker — auto-discovers master via mDNS)
python -m vramancer.main --cluster-worker

# Or configure explicitly
VRM_CLUSTER_MASTER=192.168.1.10:5000 python -m vramancer.main --cluster-worker
```

Transport selection is automatic based on locality:

| Locality | Transport | Bandwidth |
|---|---|---|
| Same GPU | Direct memory access | ∞ |
| Same node | CUDA P2P (NVLink > PCIe) | 32-900 GB/s |
| Same rack | VTP GPUDirect RDMA / RDMA verbs | 25-400 Gbps |
| Remote | VTP Zero-copy TCP | 10-100 Gbps |

---

## API Endpoints

### OpenAI-Compatible

| Endpoint | Method | Description |
|---|---|---|
| `/v1/completions` | POST | Text completion (streaming SSE supported) |
| `/v1/chat/completions` | POST | Chat completion (streaming SSE supported) |
| `/v1/batch/completions` | POST | Batch multiple prompts in one request |

### Model Management

| Endpoint | Method | Description |
|---|---|---|
| `/api/models/load` | POST | Load and split a model across GPUs |
| `/api/models` | GET | List available models |
| `/api/generate` | POST | Generate with VRAMancer-native parameters |
| `/api/infer` | POST | Raw tensor inference |

### Monitoring & Health

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Deep health check (GPU, memory, backend) |
| `/ready` | GET | Readiness probe (model loaded?) |
| `/live` | GET | Liveness probe |
| `/metrics` | GET | Prometheus metrics (47 gauges/counters/histograms) |
| `/api/pipeline/status` | GET | Pipeline state, GPU allocation, subsystem health |
| `/api/gpu` | GET | Per-GPU memory and utilization |
| `/api/system` | GET | System info (CPU, RAM, disk) |
| `/api/nodes` | GET | Cluster nodes and discovery status |
| `/api/queue/status` | GET | Request queue depth, circuit-breaker state |
| `/api/batcher/stats` | GET | Continuous batcher throughput and queue |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `VRM_API_TOKEN` | API authentication token | (required) |
| `VRM_PRODUCTION=1` | Strict mode (no security bypasses, no dev defaults) | `0` |
| `VRM_BACKEND_ALLOW_STUB=1` | Return stub if no LLM backend available | `0` |
| `VRM_MINIMAL_TEST=1` | Test mode (stubs everywhere, no GPU required) | `0` |
| `VRM_LOG_JSON=1` | Structured JSON logging | `0` |
| `VRM_TRACING=1` | OpenTelemetry tracing | `0` |
| `VRM_LEND_RATIO` | Max fraction of VRAM a GPU can lend | `0.70` |
| `VRM_RECLAIM_THRESHOLD` | VRAM utilization that triggers reclaim | `0.80` |
| `VRM_SQLITE_PATH` | Enable SQLite persistence at path | (disabled) |
| `VRM_READ_ONLY=1` | Read-only mode (no mutations) | `0` |
| `VRM_DISABLE_RATE_LIMIT=1` | Disable rate limiting (CI/testing) | `0` |

Config file: `config.yaml` (searched in XDG_CONFIG_HOME, ~/Library/Application Support, %APPDATA%, then `.`).

---

## Project Structure

```
core/                        # Production code (~80 modules)
  inference_pipeline.py      # Central orchestrator
  backends.py                # HuggingFace/vLLM/Ollama backends
  vram_lending.py            # Speculative VRAM Lending Pool
  continuous_batcher.py      # Iteration-level continuous batching
  paged_attention.py         # Multi-GPU paged KV cache + overflow lending
  model_splitter.py          # VRAM-proportional model splitting
  layer_profiler.py          # Per-layer profiling + DP-optimal placement
  transfer_manager.py        # GPU-to-GPU transport (P2P, CPU-staged, NCCL)
  production_api.py          # Flask API server (OpenAI-compatible)
  monitor.py                 # GPU monitoring + hot-plug detection
  scheduler.py               # Block allocation and migration
  hierarchical_memory.py     # 6-tier memory manager (VRAM→NVMe)
  compressor.py              # Compression (zstd/lz4) + INT4/INT8 quantization
  stream_manager.py          # Prefetch, swap, eviction scheduling
  block_router.py            # VRAM-aware routing (GPU/CPU/NVMe/network)
  compute_engine.py          # Real nn.Module execution engine
  metrics.py                 # 47 Prometheus metrics
  config.py                  # Hierarchical config (defaults→yaml→env)
  network/
    llm_transport.py         # VTP — LLM-optimized transport protocol
    fibre_fastpath.py        # RDMA / GPUDirect / Zero-copy TCP
    cluster_discovery.py     # mDNS/ZeroConf + UDP broadcast
  orchestrator/
    placement_engine.py      # Pluggable placement strategies (profiled/vram/balanced)
  security/                  # Token auth, HMAC, RBAC, rate limiting
  api/
    batch_inference.py       # Request batching engine
    circuit_breaker.py       # Circuit-breaker pattern
    validation.py            # Input validation
    registry.py              # Pipeline registry
tests/                       # ~250 tests (stub mode + real-torch CI)
monitoring/                  # Prometheus, Grafana, alerting (production-ready)
dashboard/                   # CLI + Web dashboards
vramancer/                   # CLI entry point
```

---

## Development

```bash
# Run tests (no GPU required)
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest tests/ -q --no-cov

# Lint
flake8 core/ tests/

# Build wheel
python -m build

# Smoke test
python -m tests.smoke
```

---

## Status

VRAMancer is in **beta**. Core single-node inference is production-grade; multi-node and network features are maturing.

**Stable** (tested in CI with real torch + CPU, validated with GPT-2):
- ✅ VRAM-proportional model splitting across heterogeneous GPUs
- ✅ Pipeline-parallel multi-GPU inference
- ✅ Continuous batching with iteration-level scheduling
- ✅ Multi-GPU paged KV cache with prefix caching and copy-on-write
- ✅ Speculative VRAM Lending (cooperative cross-GPU memory pooling)
- ✅ 6-tier hierarchical memory management
- ✅ OpenAI-compatible API with SSE streaming, circuit-breaker, queue management
- ✅ Full monitoring stack (47 Prometheus metrics, 24 Grafana panels, 16 alerting rules)
- ✅ Security: token auth, HMAC, JWT, RBAC, rate limiting, CORS

**Beta** (code exists, limited production validation):
- 🔶 VTP transport protocol (GPUDirect RDMA, zero-copy TCP) — requires specific hardware
- 🔶 Multi-node clustering with mDNS auto-discovery
- 🔶 GPU hot-plug detection and dynamic rebalancing

**Experimental** (proof-of-concept):
- 🔬 WebGPU browser offloading
- 🔬 USB4 hot-plug (Linux/macOS only)

### Known Limitations

- No tensor parallelism (pipeline parallelism only)
- PagedAttention operates as a memory manager (custom attention kernel for direct paged reads is planned)
- NCCL transport is reserved for multi-process distributed mode only
- JWT secret is volatile by default (lost on restart) — set `VRM_AUTH_SECRET` for persistent tokens
- Multi-GPU tests require manual execution (CI runs CPU-only real tests)
- No published performance benchmarks yet (tok/s, latency, split overhead)

---

## License

MIT

---

## ⚡ Nouvelles Fonctionnalités Expérimentales (Rust P2P & Open WebUI)

- **Contournement NVLink via PCIe (Rust) :** VRAMancer possède désormais une extension bas niveau en Rust permettant d'effectuer des transferts P2P (Peer-to-Peer) entre GPUs grand public (GeForce) sans pont NVLink en forçant le Resizable BAR (ReBAR) et l'accès DMA-BUFs direct.
- **Support OpenAI-Compatible Absolu :** API complète sur le port `5030` reprenant les endpoints officiels `/v1/models` et `/v1/chat/completions` protégés par la couche HMAC Zero-Trust.
- **Intégration Open WebUI :** Déploiement et association full support avec les dockers standards de l'industrie (ex: `open-webui/open-webui`) pour offrir une expérience "ChatGPT" inter-liée au Swarm multi-GPU. 
