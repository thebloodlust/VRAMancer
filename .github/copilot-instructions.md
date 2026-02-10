# VRAMancer — Instructions pour agents IA

## Architecture

VRAMancer est un orchestrateur multi-GPU Python (~18 000 lignes, ~200 fichiers .py) pour l'inference de modeles LLM. Le code metier vit dans `core/`, le package `vramancer/` est un wrapper d'entree (point d'entree CLI : `vramancer/main.py`).

**Flux principal :** API Flask (`core/production_api.py`) -> `InferencePipeline` (`core/inference_pipeline.py`) -> `backends.select_backend()` (factory HuggingFace/vLLM/Ollama) -> `model_splitter` (split VRAM-proportionnel) -> inference sequentielle bloc-par-bloc avec `TransferManager` (P2P/CPU-staged inter-GPU) -> reponse OpenAI-compatible.

**Pipeline d'inference (`core/inference_pipeline.py`)** — le chef d'orchestre central :
- `InferencePipeline.load(model, num_gpus)` : Backend -> Scheduler -> Monitor -> TransferManager -> load_model -> split_model -> StreamManager -> ComputeEngine
- `InferencePipeline.generate(prompt)` : tokenization -> forward multi-GPU -> decodage
- `InferencePipeline.infer(input_ids)` : forward tensor brut
- Singleton global : `get_pipeline()` / `reset_pipeline()`

**Sous-modules `core/` cles :**
- `inference_pipeline.py` — chef d'orchestre : connecte Backend, Scheduler, TransferManager, StreamManager, ComputeEngine, ClusterDiscovery, Metrics
- `backends.py` — factory LLM avec fallback stub (`VRM_BACKEND_ALLOW_STUB=1`), HuggingFaceBackend avec split_model() cable via model_splitter, generate() auto-regressif multi-GPU
- `production_api.py` — API Flask avec factory `create_app()`, endpoints OpenAI-compatible (`/v1/completions`, `/api/generate`), inference (`/api/infer`), model management (`/api/models/load`), health/ready/live, monitoring
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
- `transfer_manager.py` — transport GPU-to-GPU : CUDA P2P > CPU-staged (pinned memory). NCCL reserve au mode distribue multi-processus (MASTER_ADDR/WORLD_SIZE). Stub si `VRM_MINIMAL_TEST=1`.
- `network/fibre_fastpath.py` — transport reseau : RDMA verbs > Zero-copy TCP > mmap local
- `network/cluster_discovery.py` — mDNS/ZeroConf + UDP broadcast, heartbeat, USB4 hot-plug (pyudev/IOKit)
- `transport_factory.py` — factory unifie selon localite (SAME_GPU / SAME_NODE / SAME_RACK / REMOTE)
- `orchestrator/placement_engine.py` — placement production avec strategies pluggables (profiled/vram/balanced), DP-optimal via LayerProfiler
- `orchestrator/` — reequilibrage, migration live
- `security.py` — token + HMAC, rate limiting, RBAC ; installe via `install_security(app)`
- `metrics.py` — ~30 metriques Prometheus (port 9108)
- `api/unified_api.py` — API agregee (workflows no-code, digital twin, federated learning)
- `production_api.py` — /health (checks reels), /ready (detect_backend), /live (liveness probe), graceful shutdown SIGTERM/SIGINT

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
- **Version** : la source de verite est `core/__init__.__version__` (actuellement `0.2.4`). Les fichiers `pyproject.toml`, `setup.cfg` et `setup.py` doivent rester synchronises.
- **Config multi-OS** : `core/config.py` cherche `config.yaml` dans XDG_CONFIG_HOME (Linux), ~/Library/Application Support (macOS), %APPDATA% (Windows), puis le repertoire courant.

## Pieges connus

- `core/security/remote_access.py` lit les credentials depuis les env vars `VRM_REMOTE_ADMIN_PASS`, `VRM_REMOTE_USER_PASS`, `VRM_REMOTE_ADMIN_MFA`, `VRM_REMOTE_USER_MFA`. Sans ces variables, des valeurs dev insecurisees sont utilisees avec un warning.
- `core/security/zero_trust.py` est un stub — `authenticate()` retourne toujours `True`.
- `core/security.py` : les bypass `VRM_TEST_RELAX_SECURITY` et `VRM_TEST_BYPASS_HA` sont ignores si `VRM_PRODUCTION=1`.
- `core/auth_strong.py:ensure_default_admin()` cree un user `admin/admin` avec un warning — changer immediatement en prod.
- Les tests de `test_scheduler.py` necessitent un vrai modele GPT-2 — ils sont lents et necessitent un reseau.
- `transfer_manager.py` — transport GPU-to-GPU haute performance : CUDA P2P > CPU-staged (pinned memory). NCCL uniquement en mode distribue (MASTER_ADDR set). Supporte GPU heterogenes (Ampere + Blackwell). Stub automatique si `VRM_MINIMAL_TEST=1`.
- `network/fibre_fastpath.py` — transport reseau : RDMA verbs (ibverbs/RoCE) > Zero-copy TCP > mmap local. Support GPUDirect RDMA si nvidia_peermem charge. Classes : `RDMATransport`, `GPUDirectTransport`, `ZeroCopyTCPTransport`, `FastHandle`.
- `transport_factory.py` — factory unifie : selectionne NCCL (meme noeud) ou RDMA/TCP (reseau) selon la localite (SAME_GPU / SAME_NODE / SAME_RACK / REMOTE).
- `core/orchestrator.py` est un redirect vers `core/orchestrator/block_orchestrator.py` — ne pas y ajouter de code.

## Structure des tests

Les tests sont dans `tests/`. Le fichier `conftest.py` configure `sys.path`, definit `VRM_API_TOKEN=testtoken` et `VRM_MINIMAL_TEST=1`, mock `torch` si absent, et fournit des fixtures partages (gpu_monitor, scheduler, block_router, compressor, config, stream_manager, flask_test_client, tmp_cache_dir). Utiliser les markers pytest : `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.smoke`, `@pytest.mark.heavy`, `@pytest.mark.network`.

Le fichier `tests/test_pipeline.py` teste l'integration end-to-end : InferencePipeline, production_api (routes d'inference, health, modele), TransferManager (stub), backends, et CLI.

## Modules stubs / non fonctionnels

Ces modules existent mais ne sont pas operationnels — les traiter comme des placeholders :
- `premium/` — autotuner minimal (~30 lignes)
- `marketplace/` — squelette plugin template
- `core/cloud/hybrid_bridge.py` — stub offload cloud
- `core/simulator/digital_twin.py` — simulation basique
