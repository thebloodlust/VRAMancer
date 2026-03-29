# VRAMancer — Instructions pour agents IA

## Audit complet — 27 mars 2026

## Architecture

VRAMancer est un orchestrateur multi-GPU Python (88 fichiers .py dans core/, ~35 700 LOC core, ~204 .py total) pour l'inference de modeles LLM. Le code metier vit dans `core/`, le package `vramancer/` est un wrapper d'entree (point d'entree CLI : `vramancer/main.py`). Version courante : 1.5.0.

**Flux principal :** API Flask (`core/production_api.py`) -> `InferencePipeline` (`core/inference_pipeline.py`) -> `backends.select_backend()` (factory HuggingFace/vLLM/Ollama/llama.cpp) -> `model_splitter` (split VRAM-proportionnel) -> inference sequentielle bloc-par-bloc avec `TransferManager` (P2P/CPU-staged inter-GPU) -> reponse OpenAI-compatible.

**Pipeline d'inference (`core/inference_pipeline.py`)** — le chef d'orchestre central :
- `InferencePipeline.load(model, num_gpus)` : Backend -> Scheduler -> Monitor -> TransferManager -> load_model -> split_model -> StreamManager -> ComputeEngine
- `InferencePipeline.generate(prompt)` : tokenization -> forward multi-GPU -> decodage
- `InferencePipeline.infer(input_ids)` : forward tensor brut
- Singleton global : `get_pipeline()` / `reset_pipeline()`

## Sous-modules core/ — audit par fichier

### PRODUCTION-READY (Grade A) — code reel, teste, zero probleme majeur

| Module | LOC | Description |
|---|---|---|
| `__init__.py` | 60 | Factory stubs pour tests. Respecte VRM_MINIMAL_TEST/VRM_STRICT_IMPORT. |
| `logger.py` | 75 | JSON/Rich/File logging. Thread-local, aucun probleme. |
| `tokenizer.py` | 39 | Wrapper AutoTokenizer avec cache. |
| `metrics.py` | 200 | ~40 compteurs/gauges/histogrammes Prometheus. reset_metrics() sur shutdown. |
| `compute_engine.py` | 256 | Execution reelle nn.Module multi-accelerateur. torch.compile, ONNX export. |
| `config.py` | 440 | Resolution hierarchique multi-OS (XDG/macOS/Windows). Hot-reload (ne reinitialise pas subsystemes). |
| `model_splitter.py` | 560 | Split VRAM-proportionnel (FREE memory). Compute-aware asymetrique (Blackwell > Ampere). DP-optimal via LayerProfiler. |
| `compressor.py` | 505 | Compression reelle (zstd/lz4/gzip). INT8 quantization homebrew. INT4 mentionne mais non implemente. |
| `nvfp4_direct.py` | 290 | DirectFP4 bypass : remplace NVFP4Tensor par plain buffers + `torch._scaled_mm` direct. +7% speedup, 0 VRAM extra, match numerique exact. torch.compile compatible. |
| `auth_strong.py` | 134 | JWT + PBKDF2-HMAC SHA256. Refresh tokens rotatifs. Default admin/admin en dev (warning log). Pas de MFA. |
| `cross_vendor_bridge.py` | 620 | PipelinedTransport (double-buffer async pinned) = REEL et TESTE. DMA-BUF = STUB (pas de C extension). ReBAR detecte mais pas utilise. |
| `health.py` | 250 | Diagnostics complets GPU/RAM/disk. Pas de timeout sur pynvml. |

### FONCTIONNEL (Grade B) — fonctionne avec des limitations connues

| Module | LOC | Grade | Description | Problemes |
|---|---|---|---|---|
| `inference_pipeline.py` | 1500 | B- | Chef d'orchestre central. | Dead code WebGPU/swarm. Race condition batcher.start()/submit(). |
| `backends.py` | 2800 | B+ | HuggingFaceBackend principal. KVCacheBlock. | GPTQ monkey-patch fragile. split_model() rarement execute (accelerate prend le relais). |
| `backends_llamacpp.py` | 400 | B+ | GGUF via llama-cpp-python. Multi-GPU tensor_split. HF Hub auto-download. dp4a INT8 kernels natifs. | Aucun probleme. |
| `production_api.py` | 2000 | B+ | API Flask OpenAI-compatible. Circuit breaker. SSE streaming. | Queue depth per-process (casse en multi-worker gunicorn). SSE bypass circuit-breaker. |
| `scheduler.py` | 600 | B+ | SimpleScheduler allocation blocks. forward(), weighted_forward(). | KV-cache non comptabilise. predict_next_layers() = heuristique sequentielle naive. |
| `monitor.py` | 676 | B+ | GPUMonitor polling reel, thread-safe. ROCm-SMI fallback. | ROCm-SMI non teste sur vrai AMD. GPU index mapping assume torch.cuda = system order. |
| `continuous_batcher.py` | 1220 | B+ | Continuous batching (vLLM/Orca style). Per-request KV cache. | **Lock tenu pendant tokenizer I/O** — claim "async" est FAUX. Pas de backpressure. |
| `layer_profiler.py` | 700 | B+ | Profiling par couche (latence, FLOPS, memoire). DP placement. | Bandwidth PCIe hardcodee 25 GB/s. |
| `gpu_fault_tolerance.py` | 850 | B+ | State machine HEALTHY->DEGRADED->FAILED->OFFLINE->RECOVERING. | Probe triviale (16x16 tensor). Compteur failures ne decroit pas. |
| `hetero_config.py` | 580 | B+ | Auto-detection GPU heterogene, VM, P2P, tiers. DB 20+ cartes. | Aucun probleme. |
| `paged_attention.py` | 300+ | B+ | Block-based KV cache (vLLM-style). KV compression (turboquant). | Integration kv_quantizer pas testee en conditions reelles. |
| `benchmark.py` | 700 | B+ | 4 modes bench (tok/s, TTFT, ITL, VRAM). | Mode concurrent bypass scheduler — ne reflete pas les perfs reelles. |
| `utils.py` | 550 | B+ | detect_backend() (CUDA/ROCm/MPS/XPU/NPU/TPU). BasicTokenizer fallback. | BasicTokenizer = 70% accuracy. Mapping cache forever (pas de hotplug). |
| `speculative_decoding.py` | 380 | B | Draft model + verifier. Auto-mapping par famille (Qwen->0.5B, Llama->1B). Gamma adaptatif. | Self-drafting supprime (pas de speedup). Cable via pipeline.infer(). |
| `kv_quantizer.py` | 200+ | B | PolarQuant + QJL (~3.5 bits/dim, ~4.6x reduction). | Pas de benchmarks reels. Integration PagedKVCacheManager incertaine. |
| `transport_factory.py` | 300 | B | Factory par localite (SAME_GPU/SAME_NODE/SAME_RACK/REMOTE). | Detection topologie = string match node_id (fragile). VTP non cable. |
| `triton_sampling.py` | 200 | B | Kernel Triton fuse temperature+softmax. | top-k en Python avant kernel. Fallback PyTorch toujours utilise en pratique. |
| `tracing.py` | 75 | B | OpenTelemetry wrapper. No-op si OTEL absent ou VRM_TRACING != 1. | Aucun probleme. |
| `tensor_parallel.py` | 500 | B- | TP column/row-parallel + NCCL all-reduce. GPT-2 + Llama. | Fallback CPU casse le gradient. GQA edge cases. Teste que sur GPT-2. |
| `stream_manager.py` | 544 | B- | Prefetch, swap, eviction. Background monitoring. | Async executor jamais join() on shutdown (thread leak). Eviction LRU ignore importance. |
| `wake_on_inference.py` | 150 | B | WoL magic packets corrects. | Pas de verification de reveil. |

### INCOMPLET / EXPERIMENTAL (Grade C) — code present mais lacunaire

| Module | LOC | Grade | Description | Problemes |
|---|---|---|---|---|
| `transfer_manager.py` | 1090 | C+ | Transport GPU-to-GPU multi-strategy (0-4). | **Strategy 1.5 (Rust direct_vram_copy) = STUB** — la fonction n'existe pas encore dans vramancer_rust. Seule Strategy 4 (CPU-staged pinned) fonctionne en VM Proxmox. P2P topology cached forever. |
| `vram_lending.py` | 1000 | C+ | Lending pool cross-GPU avec lease state machine. | **Jamais teste en multi-GPU reel.** Fragmentation possible. Pas de deadlock prevention. |
| `block_router.py` | 650 | C+ | Routage VRAM-aware. RemoteExecutor. | RemoteExecutor label "zero-copy" = **FAUX** (safetensors serialise). load_block_from_disk() appelle storage_manager inexistant. |
| `backends_vllm.py` | 220 | C+ | Wrapper vLLM pass-through. | OOM retry divise max_tokens au lieu de batch_size (logique fausse). |
| `persistence.py` | 55 | C+ | SQLite CRUD workflows. | Pas de schema versioning. |
| `memory_balancer.py` | 100 | C+ | Simple LRU per GPU. | Pas de cost model migration. |
| `backends_ollama.py` | 190 | C | REST bridge sync vers Ollama. | generate_async() = dead code. aiohttp session jamais fermee (resource leak). |

### STUB / DEAD CODE (Grade D) — a supprimer ou reimplementer

| Module | LOC | Grade | Description | Problemes |
|---|---|---|---|---|
| `hierarchical_memory.py` | 1096 | B | 6-tier memory hierarchy. | eviction_cycle() + spill_to_nvme() REEL (GPU->CPU->disk via Rust cxl_direct_memory_dump). _tensor_registry tient les vrais tensors. |
| `backends_webgpu.py` | 400 | D | WebSocket WebGPU POC. | Nodes jamais peuplees. "Speculative Decoding" = batching optimiste. "Holographic Parity" = marketing. |
| `swarm_ledger.py` | 300 | D+ | Ledger SQLite complet. | Orchestrateur l'ignore. Pas de routing vers contributeurs. Fonctionnel mais orphelin. |
| `telemetry.py` | 115 | D | Format binaire custom. | Aucun consommateur. mDNS prefere. |

### Autres fichiers core/

| Module | LOC | Description |
|---|---|---|
| `cuda_graph_decode.py` | ~250 | TurboEngine : persistent CUDA Graph decode. StaticKVCache. **Incomplet** — KV update logic tronquee. |
| `triton_fused_nvfp4_quant.py` | 175 | Fused single-kernel NVFP4 activation quantizer. Triton. |
| `triton_gemv_nvfp4.py` | ~200 | Triton GEMV LUT kernel pour NVFP4. |
| `turboquant.py` | shim | Backward-compat shim vers kv_quantizer.py. |
| `holographic_memory.py` | shim | Backward-compat shim vers parity_memory.py. |
| `paged_attention_cuda.py` | 200 | Wrapper JIT pour kernel CUDA PagedAttention. Fallback PyTorch materialise KV. |
| `block_metadata.py` | ~50 | Dataclass metadata des blocks. |
| `memory_block.py` | ~50 | Dataclass memory block. |
| `memory_monitor.py` | ~100 | Monitoring memoire simple. |
| `gpu_interface.py` | ~100 | Interface GPU abstraction. |
| `model_hub.py` | ~100 | HuggingFace Hub integration. |
| `connectome.py` | ~200 | Hebbian neuroplasticity engine. **REEL** — adaptive decay, auto-calibration baseline, EMA latency. |

## core/network/ — Stack reseau AITP (8045 LOC, 19 fichiers)

### PRODUCTION-READY (Grade A)

| Module | LOC | Description |
|---|---|---|
| `network_transport.py` | 930 | **REEL** — RDMA verbs (pyverbs QP), GPUDirect RDMA (nvidia_peermem), zero-copy TCP (SO_ZEROCOPY). FastHandle auto-selection. |
| `aitp_protocol.py` | 280 | **REEL** — UDP + HMAC-SHA256 + FEC shards. hmac.compare_digest() anti-timing. send_anycast(), send_balanced(), send_raid(). IPv6 multicast ff02::vrm:1. |
| `aitp_fec.py` | 280 | **REEL** — GF(2^8) Cauchy Reed-Solomon. Tables exp/log (polynome 0x11d). Gaussian elimination avec pivoting partial. Fast-path si tous data shards presents. **Mathematiquement correct.** |
| `aitp_sensing.py` | 250 | **REEL** — IPv6 multicast heartbeat + peer tracking avec TTL eviction. HMAC auth. |
| `connectome.py` | 200 | **REEL** — Hebbian learning: adaptive decay `exp(-decay * max(0, latency - onset)) * reliability^2`. Auto-calibration baseline (median 10 premiers pings). Decay adaptatif LAN/WAN. |
| `anycast_balancer.py` | 420 | **REEL** — 3 strategies (weighted random cumsum+bisect, least_latency min(), round-robin). sync_from_connectome(), sync_from_sensing(). Failover. Prometheus metrics. |
| `network_raid.py` | 520 | **REEL** — RAID-0+RS tensor striping. ThreadPoolExecutor parallel sends. ShardReassembler avec FEC recovery automatique. JSONL journaling. |
| `cluster_discovery.py` | 750+ | **REEL** — mDNS + UDP broadcast + Bully leader election. JSONL membership. IPv6 dual-stack (multicast ff02::vrm:1 + IPv4 broadcast). getaddrinfo(AF_UNSPEC). |
| `transport.py` | 140 | **REEL** — TCP + TLS transport. |
| `security.py` | 45 | **REEL** — HMAC auth transport. |
| `edge_api.py` | 190 | **REEL** — Flask blueprint edge device lifecycle. Task inbox FIFO. |
| `fibre_fastpath.py` | 11 | Backward-compat shim vers network_transport.py. |
| `__init__.py` | 4 | Re-export passthrough. |

### FONCTIONNEL AVEC LIMITATIONS (Grade B)

| Module | LOC | Grade | Problemes |
|---|---|---|---|
| `llm_transport.py` | 1600+ | B+ | **REEL** — VTP complet : TensorHeader 64-byte, RDMA QP state machine (INIT->RTR->RTS), GPUDirect RDMA, CPU-staged, TCP fallback, KV cache streaming. VTPServer (~200 LOC) : accept_loop, _handle_client avec OOB handshake, recv_loop avec opcode routing (TENSOR/KV_CACHE/HEARTBEAT/CONTROL). **VTPServer existe et est fonctionnel** (audit precedent avait tort). |
| `aitp_receiver.py` | 350 | B+ | UDP tier fonctionne. **XDP = STUB** : `socket(44, SOCK_RAW, 0)` — famille 44 n'existe pas en Linux, toujours False. Raw socket = PermissionError sans CAP_NET_RAW. Fallback UDP solide. |
| `nat_traversal.py` | 250 | B | STUN RFC 5389 reel. UDP hole punch et relay = stubs. |
| `transmission.py` | 200 | B | Multi-protocol. UDP manque recv(). |

### INCOMPLET / STUB (Grade C-D)

| Module | LOC | Grade | Problemes |
|---|---|---|---|
| `supervision_api.py` | 540 | C+ | Flask endpoints reels. SocketIO heartbeat reel. **MAIS : NODES 100% HARDCODES FAUX** (raspberrypi, jetson, workstation fictifs). Task scheduler in-memory. HA sync = endpoint vide. |
| `webgpu_node.py` | 800 | C+ | WebSocket serveur reel. Capability negotiation reelle. **Task dispatcher incomplet.** Mock task completion auto-complete. |

## core/api/ (1091 LOC, 6 fichiers)

| Module | LOC | Grade | Description | Problemes |
|---|---|---|---|---|
| `__init__.py` | 18 | A | Re-export. | — |
| `validation.py` | 60 | A | Validation prompts + hyperparams. VRM_MAX_PROMPT_LENGTH (defaut 100K). | — |
| `circuit_breaker.py` | 179 | A- | Circuit Breaker pattern (CLOSED->OPEN->HALF_OPEN). Thread-safe. | Pas distribue (local seulement). |
| `registry.py` | 185 | B+ | PipelineRegistry singleton thread-safe pour Flask. | ClusterDiscovery auto-start dans __init__() = problematique (broadcast reseau a l'instanciation). |
| `routes_ops.py` | 321 | B- | Blueprint health/readiness/liveness/GPU/system/nodes. | **CUDA-only** : ignore ROCm/MPS dans detection GPU. |
| `batch_inference.py` | 310 | D | InferenceBatcher. | **generate_batch_fn jamais fourni** -> fallback TOUJOURS sequentiel. Dead code masque en feature. |

## core/security/ (4 fichiers)

| Module | LOC | Grade | Description |
|---|---|---|---|
| `__init__.py` | 330 | B+ | Token + HMAC-SHA256 + RBAC + rate limiting par IP+path + CSP (sans unsafe-eval). install_security(app) injecte before_request. |
| `startup_checks.py` | 65 | A | Verifications fatales au boot en mode prod : credentials par defaut, VRM_API_TOKEN requis, VRM_AUTH_SECRET requis, env vars test interdites. |
| `zero_trust.py` | shim | — | Backward-compat shim vers startup_checks.py. |

## core/orchestrator/ (4 fichiers)

| Module | LOC | Grade | Description | Problemes |
|---|---|---|---|---|
| `__init__.py` | 3 | A | Re-export PlacementEngine, BlockOrchestrator. | — |
| `placement_engine.py` | 280 | B | Strategies pluggables (profiled/vram/balanced). DP-optimal via LayerProfiler. | `_apply_neuroplasticity_score()` = heuristique pseudo-scientifique non-deterministe. |
| `heterogeneous_manager.py` | 295 | B- | Capabilities detection (cpu, gpu, ram, network). Load balancing. | GPU scoring = **string matching hardcode** (RTX 4090=+8 points). Pas de mesure reelle. |
| `block_orchestrator.py` | 184 | C+ | Block rebalancing VRAM/DRAM/NVMe/network. | Benchmarks = single-block timing (pas de contention). Network transfer appelle send_block() qui n'existe pas. |

## csrc/ — Code natif C/CUDA/C++

| Fichier | LOC | Grade | Description | Status |
|---|---|---|---|---|
| `paged_attention_kernel.cu` | 360 | A | **VRAI kernel CUDA** decode : warp-level online softmax, GQA, fp16/fp32. Un warp par (head, batch). | PRODUCTION |
| `vtp_cuda.cu` | 50 | A- | Wrapper cudaMemcpyPeerAsync propre avec CUDAGuard RAII. | PRODUCTION |
| `swarm_core.cpp` | 130 | A- | PyO3 XOR erasure coding (parity gen/repair). GIL release. | PRODUCTION |
| `aitp_xdp_bypass.c` | 170 | B | **VRAI eBPF/XDP** : parse IPv4/IPv6 -> UDP -> AITP magic -> AF_XDP redirect ou kernel DROP. | REEL mais incomplet (userspace receiver stub) |
| `rebar_mmap.c` | 200 | B- | ReBAR BAR0 detection sysfs + mmap write-combining. | REEL mais incomplet : transfer logic manquant. |
| `dmabuf_bridge.c` | 200 | C- | DMA-BUF DRM ioctl wrapper. | SQUELETTE — mmap transfer jamais implemente. |
| `software_cxl.cpp` | 70 | B- | PyO3 bindings, GIL release. | **NOM TROMPEUR** : c'est du file I/O simple (ofstream), PAS du CXL materiel. |
| `vtp_core.cpp` | 100 | C | VTP router hierarchique L1-L7. | L3+ = stub (`return src.clone()`). |

## rust_core/ — Noyau Rust haute performance

**Status : COMPILE ET FONCTIONNEL (rustc 1.75.0, cargo 1.75.0)**

| Composant | Status | Description |
|---|---|---|
| **Cargo.toml** | OK | PyO3 0.20, Tokio 1.36, cudarc (optionnel), hmac/sha2, libloading |
| **libvramancer_rust.so** | 1.6 MB (release) | Compile avec `cargo build --release --features cuda`. 1 warning trivial (unused var). |
| **CUDA FFI** | REEL | `memcpy_dtod`, `memcpy_peer_async`, `can_access_peer`, `ctx_enable_peer_access`, `mem_alloc_device/host`, `mem_free_device/host` — via libloading de libcuda.so.1. Proper error handling. |
| **HMAC validation** | REEL | Rust HMAC-SHA256 100x plus rapide que Python hmac. |
| **TransportTier** | STUB | `detect_best_transport()` retourne toujours ZeroCopyTcp. Pas de detection RDMA reelle. |
| **Triple-buffering** | REEL | Pipeline DtoD avec 3 buffers. Non cable a Python (direct_vram_copy() pas expose via PyO3). |
| **Limitation** | — | Linux only (libcuda.so.1). `.expect()` crash si libcuda absent — pas de fallback gracieux. |

## dashboard/ — Interface web et CLI

| Module | Grade | Description | Problemes |
|---|---|---|---|
| `dashboard_web.py` | C | Flask web dashboard avec GPU monitoring, model browser, chat UI. | **GPU data hardcodees dans templates.** Dependency fallback logic cassee. Demo-quality. |
| `cli_dashboard.py` | D+ | Terminal UI refresh GPU/status. | Appelle `/api/pipeline/status` qui n'existe pas dans routes_ops.py. |
| `launcher.py` | D | Entry point CLI/web. | Importe `launch_cli_dashboard()` qui **n'existe pas**. |
| `worker/` | B | matmul.wgsl + worker.js + index.html pour WebGPU compute. | Fonctionnel si navigateur supporte WebGPU. |

## Variables d'environnement essentielles

| Variable | Usage |
|---|---|
| `VRM_MINIMAL_TEST=1` | Mode test, stubs partout (pas de torch/transformers) |
| `VRM_TEST_MODE=1` | Relaxe certaines contraintes pour les tests |
| `VRM_DISABLE_RATE_LIMIT=1` | Desactive le rate limiting (CI) |
| `VRM_BACKEND_ALLOW_STUB=1` | Retourne un stub si le backend LLM est absent |
| `VRM_STRICT_IMPORT=1` | Crash si une dependance manque (defaut: degradation silencieuse) |
| `VRM_API_TOKEN` | Token d'authentification API (defaut test: `testtoken`) |
| `VRM_LOG_JSON=1` | Logs structures JSON |
| `VRM_TRACING=1` | Active OpenTelemetry |
| `VRM_SQLITE_PATH` | Active la persistence SQLite |
| `VRM_READ_ONLY=1` | Mode lecture seule (pas de mutations) |
| `VRM_PRODUCTION=1` | Mode production (validation stricte, pas de bypasses securite) |
| `VRM_CONTINUOUS_BATCHING=1` | Active le continuous batcher au chargement du modele via API |
| `VRM_GENERATE_TIMEOUT=300` | Timeout (secondes) pour les requetes via le batcher |
| `VRM_QUANTIZATION` | `nvfp4` (Blackwell FP4, CC>=10.0), `nf4` (4-bit NormalFloat, BnB), `int8` (LLM.int8, BnB), vide = BF16 |
| `VRM_KV_COMPRESSION` | `turboquant` (PolarQuant + QJL, ~3.5 bits/dim) |
| `VRM_KV_COMPRESSION_BITS` | Bits par angle polaire (defaut 3) |
| `VRM_VRAM_LENDING` | Active/desactive le VRAM Lending Pool (defaut `1` en multi-GPU) |
| `VRM_LEND_RATIO` | Ratio max de VRAM libre pretable (defaut `0.70`) |
| `VRM_RECLAIM_THRESHOLD` | Seuil d'utilisation declenchant le reclaim (defaut `0.80`) |
| `VRM_LENDING_INTERVAL` | Intervalle du monitoring lending en secondes (defaut `2.0`) |
| `VRM_TRANSFER_P2P` | Force-desactive P2P (`0`/`false`) — utile en VM IOMMU |
| `VRM_PARALLEL_MODE` | `pp` (pipeline parallel, defaut) ou `tp` (tensor parallel NCCL all-reduce) |

## Commandes de developpement

```bash
# Tests (en CI ou sans GPU)
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 pytest -q tests/

# Tests smoke rapides
python -m tests.smoke

# Lint
flake8 core/ dashboard/ tests/ || true

# Build wheel
python -m build

# Build Rust (avec CUDA FFI)
cd rust_core && cargo build --release --features cuda

# Build Rust (sans CUDA)
cd rust_core && cargo build --release
```

## Conventions et patterns

- **Imports conditionnels defensifs** : chaque module enveloppe les dependances lourdes (torch, transformers, vllm, pynvml...) dans `try/except ImportError`. Ne jamais ajouter d'import inconditionnel d'une dependance optionnelle.
- **Feature flags** : les fonctionnalites se controlent par env vars `VRM_*`, pas par fichier de config.
- **Multi-accelerateur** : `core/utils.py:detect_backend()` retourne `cuda`, `rocm`, `mps` ou `cpu`. ROCm passe par l'API `torch.cuda`. Ne jamais coder specifiquement CUDA — utiliser l'abstraction.
- **Securite Flask** : toute nouvelle route API doit passer par `install_security(app)` qui injecte un `before_request` pour validation token/HMAC.
- **Metriques** : declarer les compteurs/gauges dans `core/metrics.py` et les incrementer aux points d'instrumentation.
- **Version** : la source de verite est `core/__init__.__version__` (actuellement `1.5.0`). `pyproject.toml` et `setup.cfg` doivent rester synchronises.
- **Config multi-OS** : `core/config.py` cherche `config.yaml` dans XDG_CONFIG_HOME (Linux), ~/Library/Application Support (macOS), %APPDATA% (Windows), puis le repertoire courant.
- **Continuous batcher** : `core/continuous_batcher.py` (~1220 LOC) est cable a `generate()` dans `inference_pipeline.py`. Active via `VRM_CONTINUOUS_BATCHING=1`. Route : speculative_decoding -> batcher (si running) -> direct generate.
- **Redirects** : `core/orchestrator.py` redirige vers `core/orchestrator/block_orchestrator.py` — ne pas y ajouter de code.

## Tests — 63 fichiers, ~850+ fonctions test

```
Tests stub-safe (VRM_MINIMAL_TEST=1)  : ~750 tests (CI sans GPU)
Tests GPU reels (@pytest.mark.slow)   : ~50 tests (necessitent CUDA + modeles)
Tests integration (threading/reseau)  : ~30 tests (@pytest.mark.integration)
```

**Fichiers test cles :**
- `test_pipeline.py` — Integration end-to-end InferencePipeline
- `test_continuous_batching.py` — Batcher + PagedKVCache (~30 tests)
- `test_anycast_raid.py` — IPv6 Anycast + Network RAID (~50 tests)
- `test_aitp.py` — AITP + RS FEC (~22 tests)
- `test_rebar_lending.py` — VRAM Lending (57 tests)
- `test_nvfp4_direct.py` — DirectFP4 bypass
- `test_turboquant.py` — KV compression (23 tests)
- `test_chaos_concurrency.py` — Race condition pre-existante dans test_pipeline_concurrent_load (deselect en CI)

**Pre-existing failures :** 0-1 failures restantes (multiprocess Flask timing). 941+ tests passent.

## Benchmarks reels (23-27 mars 2026, RTX 3090 + RTX 5070 Ti, Proxmox VM)

**Single-GPU :**

| Modele | Native HF | VRAMancer | Delta |
|---|---|---|---|
| GPT-2 124M | 123.4 tok/s | 125.6 tok/s | +1.8% |
| TinyLlama-1.1B | 53.0 tok/s | 56.5 tok/s | +6.6% |
| Mistral-7B-v0.1 | 35.1 tok/s | 34.9 tok/s | -0.6% |

**THE PROOF — Heterogeneous multi-GPU :**

| Test | Resultat |
|---|---|
| Qwen2.5-14B single GPU 0 (RTX 3090 23.6 GB) | **OOM** |
| Qwen2.5-14B single GPU 1 (RTX 5070 Ti 15.5 GB) | **OOM** |
| Qwen2.5-14B VRAMancer 2-GPU | **6.0 tok/s** ✓ (GPU0: 21.7GiB/56.9%, GPU1: 14.2GiB/43.1%, 1 layer CPU overflow) |

**Quantization :**

| Modele | Quant | VRAM | Tok/s |
|---|---|---|---|
| Qwen2.5-14B | BF16 2-GPU | 35.9 GiB | 6.0 |
| Qwen2.5-14B | NF4 single GPU | 10.8 GiB | 10.5 (+75%) |
| Qwen2.5-7B | NF4 | ~5 GiB | 20.2 |
| Qwen2.5-7B GGUF Q4_K_M | llama-cpp | 3.0 GB | 106.8 (5.4x vs NF4) |

**NVFP4 Blackwell (RTX 5070 Ti) :**

| Methode | tok/s | VRAM |
|---|---|---|
| BF16 baseline | 36.4 | 15.25 GB |
| NVFP4 Dynamic W+A | 11.0 | 5.87 GB (-62% VRAM) |
| DirectFP4 bypass | 12.0 | 5.46 GB (+7% vs torchao) |

## Pieges connus — AUDIT BRUTAL

### RED FLAGS (code qui ment)

1. ~~**hierarchical_memory.py**~~ — **CORRIGE** : eviction_cycle() + spill_to_nvme() deplacent reellement les tensors (GPU->CPU->disk via Rust). _tensor_registry tient les vrais torch tensors.
2. **transfer_manager.py Strategy 1.5** — `vramancer_rust.direct_vram_copy()` **N'EXISTE PAS** dans le crate Rust. Le code Rust a le triple-buffering mais la fonction n'est pas exposee via PyO3.
3. **block_router.py RemoteExecutor** — label "zero-copy" = **FAUX** (safetensors serialise).
4. **software_cxl.cpp** — nom "CXL" = **TROMPEUR** : c'est du file I/O simple (std::ofstream).
5. **supervision_api.py** — NODES = **100% HARDCODES FAUX** (raspberrypi, jetson fictifs).
6. **batch_inference.py** — `generate_batch_fn` **JAMAIS FOURNI** -> fallback toujours sequentiel.
7. **backends_webgpu.py** — "Production Ready" = **FAUX**. POC/template.
8. **aitp_receiver.py XDP** — `socket(44, SOCK_RAW, 0)` — famille 44 invalide, toujours False. Seul UDP marche.
9. **dashboard/launcher.py** — importe `launch_cli_dashboard()` qui **N'EXISTE PAS**.
10. **placement_engine.py** — `_apply_neuroplasticity_score()` = heuristique pseudo-scientifique non-deterministe.

### LIMITATIONS REELLES

- **VM Proxmox** : Seule Strategy 4 (CPU-staged pinned) fonctionne. P2P bloque par IOMMU. Overhead VFIO ~10-15%.
- **continuous_batcher.py** : Lock GIL tenu pendant tokenizer I/O. "Async" est faux.
- **routes_ops.py** : Detection GPU **CUDA-only** — ignore ROCm/MPS.
- **BnB multi-GPU upstream bug** (accelerate 1.13.0 + BnB 0.49.2 + transformers 5.3.0) : AlignDevicesHook ne gere pas les residual connections cross-device avec couches quantifiees. VRAMancer force single-GPU pour BnB.
- **transformers 5.3 dtype** bypasse BnB : toujours utiliser torch_dtype=torch.float16 pour loads BnB.
- **auth_strong.py** : default admin/admin en dev. Changer immediatement en prod.
- **vram_lending.py** : Design ambitieux, **jamais teste en multi-GPU reel**.
- **test_chaos_concurrency.py::test_pipeline_concurrent_load** : Race condition pre-existante. Deselect en CI.

## Structure fichiers

```
core/                   88 .py  (~35,700 LOC)
  ├── api/              6 .py   (~1,090 LOC)
  ├── network/          19 .py  (~8,050 LOC)
  ├── security/         4 .py   (~460 LOC)
  └── orchestrator/     4 .py   (~760 LOC)
csrc/                   7 fichiers C/CUDA/C++
rust_core/              Cargo crate (PyO3 + Tokio + CUDA FFI) — 1.6 MB .so
tests/                  63 fichiers — ~850+ tests
benchmarks/             24 fichiers
dashboard/              5 .py + templates/ + static/ + worker/
vramancer/              CLI wrapper (main.py + __main__.py + cli/)
_deprecated/            16 fichiers (backends_deepspeed, backends_tensorrt, network_archive, etc.)
monitoring/             5 fichiers (Prometheus/Grafana/alerting)
examples/               4 fichiers
scripts/                9 fichiers
```

## Resume honnetete globale

| Categorie | Fichiers | % reel |
|---|---|---|
| PRODUCTION (Grade A) | 12 core + 10 network + 4 api/security | ~30% des fichiers, ~40% du code |
| FONCTIONNEL (Grade B) | 20 core + 4 network | ~30% des fichiers, ~40% du code |
| INCOMPLET (Grade C) | 7 core + 2 network + 2 orchestrator | ~15% des fichiers, ~10% du code |
| STUB/DEAD (Grade D) | 4 core + 1 api + 2 dashboard | ~10% des fichiers, ~10% du code |
| Rust crate | 1 crate, compile | CUDA FFI reel, transport stub, triple-buffer non cable |
| C/CUDA natif | 7 fichiers | kernel CUDA reel (A), eBPF reel (B), reste incomplet |

**Verdict : ~70% du code est reel et fonctionnel. ~15% est incomplet mais avec du vrai code derriere. ~15% est du stub/dead code/marketing.**
