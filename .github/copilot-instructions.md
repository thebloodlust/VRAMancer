# VRAMancer — Instructions pour agents IA

## Architecture

VRAMancer est un orchestrateur multi-GPU Python (~30 000 lignes, ~80 fichiers .py dans core/) pour l'inference de modeles LLM. Le code metier vit dans `core/`, le package `vramancer/` est un wrapper d'entree (point d'entree CLI : `vramancer/main.py`). Version courante : 1.5.0.

**Flux principal :** API Flask (`core/production_api.py`) -> `InferencePipeline` (`core/inference_pipeline.py`) -> `backends.select_backend()` (factory HuggingFace/vLLM/Ollama/llama.cpp) -> `model_splitter` (split VRAM-proportionnel) -> inference sequentielle bloc-par-bloc avec `TransferManager` (P2P/CPU-staged inter-GPU) -> reponse OpenAI-compatible.

**Pipeline d'inference (`core/inference_pipeline.py`)** — le chef d'orchestre central :
- `InferencePipeline.load(model, num_gpus)` : Backend -> Scheduler -> Monitor -> TransferManager -> load_model -> split_model -> StreamManager -> ComputeEngine
- `InferencePipeline.generate(prompt)` : tokenization -> forward multi-GPU -> decodage
- `InferencePipeline.infer(input_ids)` : forward tensor brut
- Singleton global : `get_pipeline()` / `reset_pipeline()`

**Sous-modules `core/` cles :**
- `inference_pipeline.py` — chef d'orchestre : connecte Backend, Scheduler, TransferManager, StreamManager, ComputeEngine, ClusterDiscovery, Metrics
- `backends.py` — factory LLM avec fallback stub (`VRM_BACKEND_ALLOW_STUB=1`), HuggingFaceBackend avec split_model() cable via model_splitter, generate() auto-regressif multi-GPU. vLLMBackend, OllamaBackend et LlamaCppBackend extraits dans `backends_vllm.py`, `backends_ollama.py` et `backends_llamacpp.py`.
- `production_api.py` — API Flask avec factory `create_app()`, endpoints OpenAI-compatible (`/v1/completions`, `/api/generate`), inference (`/api/infer`), model management (`/api/models/load`). Routes ops/health extraites dans `core/api/routes_ops.py`.
- `scheduler.py` — SimpleScheduler avec allocate_block/release_block/predict_next_layers/find_alternate_gpu/migrate_block + forward/predict
- `block_router.py` — routage VRAM-aware vers GPU/CPU/NVMe/reseau avec detection NVMe reelle et registre dynamique de noeuds
- `monitor.py` — GPUMonitor production avec vram_usage(), detect_overload(), polling background, export Prometheus, ROCm-SMI fallback
- `stream_manager.py` — StreamManager production : prefetch, swap, eviction, monitoring background
- `compressor.py` — compression reelle (zstd/lz4/gzip), quantization INT8/INT4, compress_tensor/compress_module
- `hierarchical_memory.py` — 6 niveaux (VRAM->DRAM->NVMe->reseau) avec scoring LRU/LFU hybride
- `config.py` — resolution hierarchique : defaults -> config.yaml (multi-OS) -> env vars VRM_*, validation, hot-reload
- `compute_engine.py` — execution de vrais modules nn.Module (pas de poids aleatoires)
- `model_splitter.py` — split VRAM-proportionnel (FREE memory) ou profiler-optimise (DP), support MPS, imports defensifs
- `layer_profiler.py` — profiling par couche (latence, FLOPS, memoire), benchmark GPU (compute, bandwidth), placement DP-optimal
- `nvfp4_direct.py` — DirectFP4 bypass : remplace les NVFP4Tensor (torchao) par des plain buffers + appel direct `torch._scaled_mm`, eliminant le overhead `__torch_dispatch__` (~7% speedup, 0 VRAM extra). Active automatiquement apres quantization dans `_apply_nvfp4_quantization()`.
- `transfer_manager.py` — transport GPU-to-GPU : Strategy 0 (cross-vendor bridge) > Strategy 1 (CUDA P2P) > Strategy 1.5 (optional Rust cuMemcpyDtoD bypass via `pip install vramancer_rust[cuda]`, ~1.3-1.6x vs PyTorch .to()) > Strategy 2 (ReBAR pipelined) > Strategy 3 (NCCL distribue) > Strategy 4 (CPU-staged pinned). En VM Proxmox, seule Strategy 4 fonctionne. Stub si `VRM_MINIMAL_TEST=1`.
- `network/fibre_fastpath.py` — transport reseau : RDMA verbs (ibverbs/RoCE) > Zero-copy TCP > mmap local. Support GPUDirect RDMA si nvidia_peermem charge. DMA-BUF = stub (pas de C extension).
- `transport_factory.py` — factory unifie : selectionne NCCL (meme noeud) ou RDMA/TCP (reseau) selon la localite (SAME_GPU / SAME_NODE / SAME_RACK / REMOTE).
- `core/orchestrator.py` est un redirect vers `core/orchestrator/block_orchestrator.py` — ne pas y ajouter de code.
- `api/unified_api.py` — **SUPPRIME** : reference dans 9 fichiers docs/ mais n'existe plus. Les docs sont obsoletes.

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
| `VRM_QUANTIZATION` | Quantization mode: `nvfp4` (Blackwell native FP4, CC>=10.0, torchao), `nf4` (4-bit NormalFloat, BnB), `int8` (LLM.int8, BnB), vide = BF16. `nvfp4` auto-fallback vers `nf4` si pas de GPU Blackwell ou torchao absent. |

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
```

## Conventions et patterns

- **Imports conditionnels defensifs** : chaque module enveloppe les dependances lourdes (torch, transformers, vllm, pynvml...) dans `try/except ImportError`. Ne jamais ajouter d'import inconditionnel d'une dependance optionnelle.
- **Feature flags** : les fonctionnalites se controlent par env vars `VRM_*`, pas par fichier de config.
- **Multi-accelerateur** : `core/utils.py:detect_backend()` retourne `cuda`, `rocm`, `mps` ou `cpu`. ROCm passe par l'API `torch.cuda`. Ne jamais coder specifiquement CUDA — utiliser l'abstraction.
- **Securite Flask** : toute nouvelle route API doit passer par `install_security(app)` qui injecte un `before_request` pour validation token/HMAC.
- **Metriques** : declarer les compteurs/gauges dans `core/metrics.py` et les incrementer aux points d'instrumentation appropries.
- **Version** : la source de verite est `core/__init__.__version__` (actuellement `1.5.0`). Les fichiers `pyproject.toml`, `setup.cfg` et `setup.py` doivent rester synchronises.
- **Config multi-OS** : `core/config.py` cherche `config.yaml` dans XDG_CONFIG_HOME (Linux), ~/Library/Application Support (macOS), %APPDATA% (Windows), puis le repertoire courant.
- **Continuous batcher** : `core/continuous_batcher.py` (~700 LOC) est cable a `generate()` dans `inference_pipeline.py`. Active via `VRM_CONTINUOUS_BATCHING=1` au chargement API. Route : speculative_decoding -> batcher (si running) -> direct generate.

## Pieges connus

- `core/security/remote_access.py` — **deplace vers `_deprecated/`**. Module orphelin, l'app Flask port 5001 n'etait jamais lancee.
- `core/security/zero_trust.py` est un shim backward-compat, redirige vers startup_checks.py.
- `core/security/__init__.py` : les bypass `VRM_TEST_RELAX_SECURITY` et `VRM_TEST_BYPASS_HA` sont ignores si `VRM_PRODUCTION=1`. CSP corrige : whitelist CDN specifiques, `unsafe-inline` conserve (templates), `unsafe-eval` **supprime**.
- `core/auth_strong.py:ensure_default_admin()` cree un user `admin/admin` avec un warning — changer immediatement en prod.
- Les tests de `test_scheduler.py` necessitent un vrai modele GPT-2 — ils sont lents et necessitent un reseau.
- `transfer_manager.py` — En VM Proxmox, seule la Strategy 4 (CPU-staged pinned) fonctionne. Strategy 1.5 (Rust bypass) necessite `pip install vramancer_rust[cuda]` (optionnel, ~1.3-1.6x vs PyTorch .to()). NCCL uniquement en mode distribue (MASTER_ADDR set).
- `block_router.py` — RemoteExecutor utilise safetensors serialization pour les transfers reseau. Labels "zero-copy" **corriges** en "binary serialization".
- `hierarchical_memory.py` — NVMe spill via Rust direct I/O (vramancer_rust) ou FastNVMe fallback. Labels "CXL" **corriges** en "Direct I/O". software_cxl fallback **supprime**.
- `network/fibre_fastpath.py` — transport reseau : RDMA verbs (ibverbs/RoCE) > Zero-copy TCP > mmap local. Support GPUDirect RDMA si nvidia_peermem charge. DMA-BUF = stub (pas de C extension).
- `transport_factory.py` — factory unifie : selectionne NCCL (meme noeud) ou RDMA/TCP (reseau) selon la localite (SAME_GPU / SAME_NODE / SAME_RACK / REMOTE).
- `core/orchestrator.py` est un redirect vers `core/orchestrator/block_orchestrator.py` — ne pas y ajouter de code.
- `api/unified_api.py` — **SUPPRIME** : reference dans 9 fichiers docs/ mais n'existe plus. Les docs sont obsoletes.
- `api/circuit_breaker.py` — bien concu, instancie par production_api.py (create_app), protege `_run_with_timeout()` et `_guarded_sse()`.
- `api/batch_inference.py` — `generate_batch_fn` jamais fourni, fallback toujours sequentiel.
- `triton_sampling.py` — kernel Triton corrige : top-k applique en Python avant le kernel, Triton fuse temperature+softmax. Actif quand `top_p >= 1.0` (cas courant).
- `dashboard/cli_dashboard.py` — appelle `/api/gpu` et `/api/status` qui n'existent pas dans production_api.py.
- `dashboard/dashboard_web.py` — donnees GPU hardcodees, swarm status bidon, security fallback no-op.
- `network/interface_selector.py` — **corrige** (import psutil fixe). `network/resource_aggregator.py` — **deplace vers `_deprecated/`**. `network/security.py` — **corrige** (logging fixe).
- `speculative_decoding.py` — cable au pipeline via `self.infer()`. Auto-mapping draft models par famille (Qwen→0.5B, Llama→1B). Self-drafting supprime.
- `backends_webgpu.py` — POC/template. Nodes jamais peuplees. "Speculative Decoding" = batching optimiste sans verification. Pas un vrai backend.
- **BnB multi-GPU upstream bug** (accelerate 1.13.0 + BnB 0.49.2 + transformers 5.3.0) : `AlignDevicesHook` ne gere pas les residual connections cross-device avec des couches quantifiees. VRAMancer force single-GPU pour les loads BnB. INT8 14B ne tient pas sur un seul 24 GB GPU.
- **transformers 5.3 `dtype` bypasse BnB** : le nouveau parametre `dtype` ignore completement la quantification BnB (charge en full precision). Toujours utiliser `torch_dtype=torch.float16` pour les loads BnB.

## Maturite des modules (audit ultra-honnete — 23 mars 2026)

### PRODUCTION — code reel, teste, cable dans le pipeline (32 modules)

| Module | LOC | Note | Remarques |
|---|---|---|---|
| `inference_pipeline.py` | 1500 | A- | Chef d'orchestre central. Code WebGPU/swarm mort a nettoyer. Race condition batcher.start()/submit(). |
| `backends.py` (HF) | 2800 | B+ | Backend principal. split_model() rarement execute (accelerate prend le relais). GPTQ monkey-patch fragile. |
| `production_api.py` | 2000 | B+ | API Flask OpenAI-compatible. Queue depth par process (casse en multi-worker). SSE bypass circuit-breaker. |
| `scheduler.py` | 600 | B+ | Allocation blocks OK. KV-cache non comptabilise. migrate_block() utilise TransferManager avec fallback .to(). predict_next_layers() = heuristique sequentielle naive. |
| `model_splitter.py` | 560 | A | Split VRAM-proportionnel correct. Compute-aware. |
| `compute_engine.py` | 256 | A | Execution reelle nn.Module multi-accelerateur. |
| `monitor.py` | 676 | B+ | Polling reel, thread-safe. ROCm-SMI non teste sur vrai AMD. |
| `continuous_batcher.py` | 1220 | B+ | Batched/chunked prefill. Lock encore tenu pendant tokenizer (claim async faux). |
| `paged_attention.py` | 1100 | B | Pool layout correct. VRAM lending integration non testee. |
| `compressor.py` | 505 | A- | zstd/lz4/gzip reel. INT4 quantization homebrew non verifiee. |
| `config.py` | 440 | A | Resolution hierarchique multi-OS. Hot-reload ne reinitialise pas les subsystemes. |
| `utils.py` | 550 | B+ | Detection GPU multi-accelerateur. Mapping cache forever (pas de hotplug). BasicTokenizer = 70% accuracy. |
| `transfer_manager.py` | 1090 | C+ | Strategy 4 (CPU-staged) seule fonctionnelle en VM. Strategy 1.5 (Rust) = stub (vramancer_rust.direct_vram_copy n'existe pas). |
| `stream_manager.py` | 544 | B- | Prefetch/swap/eviction OK. Queue async leak a l'arret. |
| `block_router.py` | 650 | C+ | Routage VRAM-aware OK. RemoteExecutor "zero-copy" = FAUX (safetensors serialise). vramancer_rust.send_tensor_p2p inexistant. |
| `cross_vendor_bridge.py` | 1000 | B- | PipelinedTransport (double-buffer async) = REEL et TESTE. DMA-BUF = STUB (pas de C extension). ReBAR detecte mais jamais utilise. |
| `layer_profiler.py` | 700 | A- | Profiling reel torch.mm(). DP placement. Bandwidth PCIe hardcodee 25 GB/s. |
| `metrics.py` | 140 | B | ~35 metriques Prometheus. WebGPU metrics cassees. Pas de lifecycle management. |
| `health.py` | 250 | A- | Diagnostics complets. Pas de timeout sur pynvml. |
| `gpu_fault_tolerance.py` | 850 | B+ | State machine HEALTHY→DEGRADED→FAILED→OFFLINE. Probe triviale (16×16 tensor). |
| `transport_factory.py` | 300 | B | Factory unifiee par localite. Detection topologie primitive (string match node_id). |
| `auth_strong.py` | 134 | A | JWT + PBKDF2-HMAC SHA256 correct. Pas de MFA. |
| `hetero_config.py` | 580 | A | Auto-detection GPU heterogene, VM, P2P, tiers, lending policy. DB 20+ cartes. |
| `logger.py` | 75 | A | JSON/Rich/File. Aucun probleme. |
| `security/__init__.py` | 640 | B+ | Token + HMAC + RBAC + rate limiting. CSP corrige (whitelist CDN, sans unsafe-eval). |
| `api/registry.py` | 145 | B | PipelineRegistry. ClusterDiscovery auto-start dans __init__ = problematique. |
| `api/routes_ops.py` | 321 | B- | Routes health/gpu. Detection GPU CUDA-only (ignore ROCm/MPS). |
| `placement_engine.py` | 280 | B+ | Strategies pluggables, DP-optimal. Neuroplasticity score non documente. Cache GPU sans TTL. |
| `cluster_discovery.py` | 515 | A- | mDNS + UDP broadcast + Bully leader election. JSONL membership. |
| `fibre_fastpath.py` | 620 | A- | RDMA verbs reel (pyverbs QP), GPUDirect RDMA, zero-copy TCP. |
| `nat_traversal.py` | 320 | A | STUN RFC 5389, UDP hole punch, relay, ULA IPv6. |
| `supervision_api.py` | 550 | B+ | Flask, task scheduler, HA delta sync zstd/lz4/zlib. |

### FONCTIONNEL — code qui marche mais avec des lacunes (20 modules)

| Module | LOC | Note | Remarques |
|---|---|---|---|
| `vram_lending.py` | 1000 | C+ | Design ambitieux (lease state machine, scoring, reclaim). **Jamais teste en multi-GPU reel**. Fragmentation possible. |
| `paged_attention_cuda.py` | 200 | B | Wrapper JIT CUDA. Fallback PyTorch materialise KV (defait le paging). Kernel CUDA 8.8x@ctx64. |
| `tensor_parallel.py` | 500 | B- | TP NCCL all-reduce. Teste sur GPT-2. Fallback CPU casse le gradient. GQA edge cases. |
| `triton_sampling.py` | 200 | B | Kernel Triton fuse temperature+softmax. top-k applique en Python avant kernel. Actif quand top_p >= 1.0 (cas courant). |
| `hierarchical_memory.py` | 1096 | D+ | Registre metadata seulement. CXL bridge = STUB (software_cxl inexistant). _evict_lru() log "physically offloaded" mais ne deplace rien. |
| `backends_vllm.py` | 220 | C+ | Wrapper passe-plat. Retry OOM divise max_tokens (devrait diviser batch_size). |
| `backends_ollama.py` | 190 | C | REST bridge sync OK. async generate_async() = DEAD CODE. Session aiohttp jamais fermee. |
| `backends_llamacpp.py` | 400 | B+ | GGUF via llama-cpp-python. Multi-GPU tensor_split, HF Hub auto-download (Q4_K_M prefere). dp4a INT8 kernels natifs. Stub mode OK. |
| `backends_webgpu.py` | 400 | D | Template/POC. Nodes jamais peuplees. "Speculative Decoding" = batching optimiste sans verification. "Holographic Parity" = marketing pur. |
| `benchmark.py` | 700 | C+ | 4 modes. Mode concurrent bypasse scheduler. Ne reflete pas les perfs multi-GPU reelles. |
| `api/batch_inference.py` | 309 | D | `generate_batch_fn` jamais fourni → fallback sequentiel. Dead code masque en feature. |
| `api/circuit_breaker.py` | 165 | B+ | Instancie dans create_app(). Protege _run_with_timeout() et _guarded_sse(). |
| `api/validation.py` | 60 | B+ | Valide hyperparams + prompt length (VRM_MAX_PROMPT_LENGTH, defaut 100K chars). |
| `block_orchestrator.py` | 155 | C | Benchmarking fake (single-op timing). Logique reseau incomplete. Metrics silencieusement ignorees. |
| `heterogeneous_manager.py` | 400 | C+ | Scores GPU hardcodes (RTX 4090=+8 points). String-based detection. Load balancing naif. |
| `memory_balancer.py` | 100 | C | Simple LRU. Pas de cost model migration. |
| `persistence.py` | 55 | C+ | SQLite CRUD. Pas de schema versioning. |
| `connectome.py` | 200 | B | Hebbian synapse weights. EMA + exponential decay. Correct. |
| `wake_on_inference.py` | 150 | C | WoL magic packets corrects. Pas de verification de reveil. |
| `edge_api.py` | 180 | B | Flask blueprint edge device lifecycle. Task inbox FIFO. |
| `aitp_fec.py` | 270 | B- | **Vrai** GF(2^8) Cauchy Reed-Solomon (PAS du XOR simple — corrige par rapport a l'ancien audit). BUG: decode() fast-path echoue sur indices non-contigus. |

### STUBS / DEAD CODE — a supprimer ou reimplementer (15 modules)

| Module | LOC | Statut | Raison |
|---|---|---|---|
| `speculative_decoding.py` | 380 | CABLE | Algorithme correct. `swarm_verify_callable` = `pipeline.infer()`. Auto-mapping draft models par famille (Qwen→0.5B, Llama→1B). Self-drafting supprime (pas de speedup). |
| `holographic_memory.py` | 190 | RENOMME | Renomme en `core/parity_memory.py` (XOR parity, pas "holographique"). `_deprecated/holographic_memory.py` redirige. |
| `swarm_ledger.py` | 300 | DECONNECTE | Ledger SQLite complet mais orchestrateur l'ignore. Pas de routing vers contributeurs. |
| `telemetry.py` | 115 | INUTILISE | Format binaire custom. Aucun consommateur. mDNS prefere. |
| `network/interface_selector.py` | 35 | **CASSE** | Import psutil avant def. Fonctions appelees avant definition. |
| `network/packets.py` | 25 | REMPLACE | Remplace par transmission.py inline. |
| `network/resource_aggregator.py` | 30 | **CASSE** | Appelle `create_local_cluster()` inexistant. |
| `network/security.py` | 30 | **CASSE** | `logging.info(..., file=sys.stderr)` invalide. |
| `network/vramancer_link.py` | 5 | INUTILE | Re-export 2 lignes, jamais importe. |
| `network/llm_transport.py` | 1100 | INCOMPLET | VTP coupe a 60%. VTPServer manquant. Claim "Production Ready" = FAUX. |
| `network/webgpu_node.py` | 800 | ORPHELIN | Prototype WebSocket. hive_memory.encode_hologram() non lie. Jamais instancie. |
| `security/remote_access.py` | 198 | ORPHELIN | Flask app port 5001 jamais lancee. MFA statique. |
| `core/__init__.py` | 60 | DEAD | Factory de stubs pour tests. Production importe directement les sous-modules. |
| `dashboard/cli_dashboard.py` | 60 | **CASSE** | Appelle `/api/gpu` et `/api/status` qui n'existent pas. |
| `dashboard/dashboard_web.py` | 250 | INCOMPLET | Donnees GPU hardcodees fakes. Swarm status bidon. Security fallback no-op. |

### Nouveaux modules (session 23 mars 2026)

| Module | LOC | Statut | Description |
|---|---|---|---|
| `csrc/paged_attention_kernel.cu` | 360 | PRODUCTION | Kernel CUDA decode : warp-level online softmax, GQA, fp16/fp32. 8.8x@ctx64. |
| `core/paged_attention_cuda.py` | 200 | FONCTIONNEL | Wrapper JIT Python. Auto-detection arch nvcc. Fallback PyTorch. |
| `core/tensor_parallel.py` | 500 | FONCTIONNEL | TP column/row-parallel + NCCL all-reduce. GPT-2 + Llama. |
| `core/triton_sampling.py` | 200 | STUB | Kernel Triton jamais appele en pratique. |
| `core/nvfp4_direct.py` | 290 | PRODUCTION | DirectFP4 bypass: remplace NVFP4Tensor par plain buffers + `_scaled_mm` direct. 1.07x vs torchao, 0 VRAM extra. torch.compile compatible. |

### Benchmarks reels (23-24 mars 2026, RTX 3090 + RTX 5070 Ti, Proxmox VM)

**Single-GPU (bench_tok_s.py) :**

| Modele | Params | Native HF | VRAMancer | Delta |
|---|---|---|---|---|
| GPT-2 | 124M | 123.4 tok/s | 125.6 tok/s | **+1.8%** |
| TinyLlama-1.1B | 1.1B | 53.0 tok/s | 56.5 tok/s | **+6.6%** |
| Mistral-7B-v0.1 | 7.2B | 35.1 tok/s | 34.9 tok/s | -0.6% |
| GPT-2 multi-GPU (PP) | 124M | 124.6 tok/s (1 GPU) | 92.0 tok/s (2 GPU) | -26.2% (model trop petit) |
| GPT-2 multi-GPU (TP) | 124M | 124.6 tok/s (1 GPU) | 86.0 tok/s (2 GPU) | -31.0% (model trop petit) |
| vs vLLM | — | NON TESTE | — | vLLM downgrade transformers 5.3→4.57 |

**Heterogeneous multi-GPU — THE PROOF (bench_heterogeneous.py) :**

| Test | GPU(s) | Resultat |
|---|---|---|
| Qwen2.5-14B-Instruct single GPU 0 | RTX 3090 (23.6 GB) | **OOM** |
| Qwen2.5-14B-Instruct single GPU 1 | RTX 5070 Ti (15.5 GB) | **OOM** |
| Qwen2.5-14B-Instruct VRAMancer 2-GPU | 3090 + 5070 Ti (38.5 GB pool) | **6.0 tok/s** ✓ |

Le modele 14B (28GB bf16) ne tient sur aucun GPU individuel. VRAMancer le repartit automatiquement
(GPU 0: 21.7GiB/56.9%, GPU 1: 14.2GiB/43.1%, 1 layer CPU overflow — lm_head). Layers: {cuda:0: 23, cuda:1: 28, cpu: 1}.
Chargement en 3.9s, generation coherente verifiee. Budget VRAM agressif (92%) pour minimiser le CPU offload.

**Quantization tiers (bench_heterogeneous.py --quantization) :**

| Modele | Quant | GPUs | VRAM | Tok/s | vs BF16 14B |
|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct | BF16 | 2-GPU | 35.9 GiB | 6.0 | baseline |
| Qwen2.5-14B-Instruct | **NF4** | single GPU 0 | 10.8 GiB | **10.5** | **+75%** |
| Qwen2.5-7B-Instruct | NF4 | single GPU 0 | ~5 GiB | **20.2** | — |
| Qwen2.5-7B-Instruct | INT8 | single GPU 0 | ~10 GiB | **8.1** | — |
| Qwen2.5-14B-Instruct | INT8 | — | — | OOM | limitation |

NF4 est **75% plus rapide** que BF16 pour le 14B car le modele tient sur un seul GPU (10.8 GiB vs 28 GiB), eliminant le overhead de transfert inter-GPU. INT8 14B ne tient pas sur un seul GPU et le multi-GPU BnB est casse upstream (accelerate 1.13.0 + BnB 0.49.2 + transformers 5.3.0).

**GGUF llama.cpp (Qwen2.5-7B-Instruct Q4_K_M) :**

| Methode | tok/s | TTFT | VRAM | Taille modele |
|---|---|---|---|---|
| GGUF Q4_K_M (llama-cpp-python, 1-GPU) | **106.8** | 27 ms | 3.0 GB | 4.4 GB |
| GGUF Q4_K_M (VRAMancer LlamaCppBackend) | **106.7** | 15 ms | 3.0 GB | 4.4 GB |
| BnB NF4 (HuggingFace generate) | 19.7 | — | 6.6 GB | ~5 GB |

GGUF Q4_K_M est **5.4x plus rapide** que BnB NF4 grace aux kernels dp4a INT8 (poids restent quantifies pendant le calcul, 4 MACs/cycle vs fp16 1 MAC/cycle). GGUF utilise aussi 2.2x moins de VRAM et charge 5.7x plus vite.

**NVFP4 Blackwell (Qwen2.5-7B-Instruct, RTX 5070 Ti CC 12.0) :**

| Methode | tok/s | VRAM | vs BF16 |
|---|---|---|---|
| BF16 (baseline) | 36.4 | 15.25 GB | 100% |
| NVFP4 Dynamic W+A (cublas FP4) | **11.0** | **5.87 GB** | 30% |
| BnB NF4 | 17.5 | 14.77 GB | 48% |
| BnB INT8 | 7.9 | 14.76 GB | 22% |

NVFP4 Dynamic W+A utilise le vrai kernel cublas Blackwell FP4 (`torch._scaled_mm` avec `float4_e2m1fn_x2`). Economise 62% VRAM vs BF16 mais plus lent que BnB NF4 — torchao 0.16.0 est encore **prototype** (overhead Python dispatch + quantization dynamique des activations). NVFP4 Weight-Only inutilisable (0.9 tok/s — dequantise a chaque pass). `lm_head` exclu de la quantization (aten.expand non implemente pour NVFP4Tensor).

**DirectFP4 Bypass (Qwen2.5-7B-Instruct, RTX 5070 Ti) :**

| Methode | tok/s | VRAM | vs torchao |
|---|---|---|---|
| torchao NVFP4 (baseline) | 11.2 | 5.46 GB | 1.00x |
| **DirectFP4 bypass** | **12.0** | **5.46 GB** | **1.07x** |
| DirectFP4 + torch.compile | 12.0 | 5.46 GB | 1.07x |

DirectFP4 (`core/nvfp4_direct.py`) remplace les NVFP4Tensor par des plain buffers + appel direct `torch._scaled_mm`, eliminant le overhead `__torch_dispatch__`. 7% speedup, 0 VRAM extra, match numerique exact avec torchao. torch.compile fonctionne (bloque sur NVFP4Tensor). Active automatiquement dans `_apply_nvfp4_quantization()`.

**Contexte VM :** Proxmox VFIO passthrough, P2P bloque (IOMMU), CPU-staged transfers ~11 GB/s, overhead VFIO ~10-15% sur PCIe.
**Bare metal attendu :** +10-30% sur les transfers si P2P/NVLink disponible.

### Tests : 764 passed, 34 skipped, 0 failed (56 fichiers test)

## Structure des tests

Les tests sont dans `tests/`. Le fichier `conftest.py` configure `sys.path`, definit `VRM_API_TOKEN=testtoken` et `VRM_MINIMAL_TEST=1`, mock `torch` si absent, et fournit des fixtures partages (gpu_monitor, scheduler, block_router, compressor, config, stream_manager, flask_test_client, tmp_cache_dir). Utiliser les markers pytest : `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.smoke`, `@pytest.mark.heavy`, `@pytest.mark.network`.

Le fichier `tests/test_pipeline.py` teste l'integration end-to-end : InferencePipeline, production_api (routes d'inference, health, modele), TransferManager (stub), backends, et CLI.

## Modules extraits / architecture

- `core/backends.py` — contient BaseLLMBackend, HuggingFaceBackend, KVCacheBlock, select_backend()
- `core/backends_vllm.py` — vLLMBackend (extrait de backends.py)
- `core/backends_ollama.py` — OllamaBackend (extrait de backends.py)
- `core/backends_llamacpp.py` — LlamaCppBackend (GGUF via llama-cpp-python, multi-GPU tensor_split, HF Hub auto-download)
- `core/backends_webgpu.py` — WebGPU offloading via WebSocket
- `_deprecated/backends_deepspeed.py` — DeepSpeed (non integre, archive)
- `_deprecated/backends_tensorrt.py` — TensorRT (non integre, archive)
- `core/api/routes_ops.py` — routes health/system/gpu (Blueprint extrait de production_api.py)
- `core/api/registry.py` — PipelineRegistry (extrait de production_api.py)
- `core/api/validation.py` — validation des parametres (extrait de production_api.py)
- `core/security/startup_checks.py` — verifications securite au demarrage (remplace zero_trust.py)
- `core/security/zero_trust.py` — shim backward-compat, redirige vers startup_checks.py
- `csrc/paged_attention_kernel.cu` — kernel CUDA PagedAttention decode (warp-level softmax, GQA, fp16/fp32)
- `core/paged_attention_cuda.py` — wrapper JIT + fallback PyTorch pour le kernel CUDA
- `core/tensor_parallel.py` — TP column/row-parallel + NCCL all-reduce (GPT-2 + Llama)
