# VRAMancer — TODO (refresh 2026-05-10, post V6.E)

État courant : 1.5.0, branche `feat/v6-lending-cooperative`.
~70% du code est réel et fonctionnel, ~15% incomplet, ~15% stub/dead code.
Légende effort : **(S)** <1h • **(M)** 1–4h • **(L)** >4h • **(XL)** plusieurs jours

---

## ✅ Récemment livré (session 2026-05-10)

Audit cleanup + auto-detection + Rust hardening :

- [x] §audit #2 WebGPU "Production Ready" → "experimental — POC, not production-ready" (`core/webgpu_backend.py`)
- [x] §audit #3 `swarm_ledger.py` orphelin → redirect explicite vers `_deprecated/`
- [x] §audit #4 `supervision_api.py` NODES hardcodés → `NODES = []` dynamique (heartbeat/discovery)
- [x] §audit #5 docstrings honnêtes : `cross_vendor_bridge.py` plus de "true zero-copy" trompeur, csrc DMA-BUF + `file_offload.cpp` (ex `software_cxl.cpp`) headers explicites
- [x] §audit #6 centralisation env vars : `core/env_flags.py` registry de **140 flags** + `dump_flags()` / `dump_active()` / `unknown_env_flags()` / CLI debug
- [x] §1 SQLite schema versioning v2 (`core/persistence.py`, `schema_version` table + `created_at`)
- [x] §2 `recommend_quantization()` SM/VRAM → nvfp4/bf16/nf4/int8/gguf (`core/auto_detect.py`)
- [x] §2 `VRM_QUANTIZATION=auto` câblé dans HuggingFaceBackend
- [x] §2 `detect_virtualization()` + `should_disable_p2p()` (Proxmox Q35/i440FX, KVM, VMware, Hyper-V) câblé dans `TransferManager.__init__`
- [x] §5 Rust crate : `cuda_available()` exposé + chargement `Option<libloading::Library>` (plus de `.expect` qui crash) + support `nvcuda.dll` Windows
- [x] §5 `core/rust_bridge.py` : façade Python qui ne lève jamais d'exception (`has_rust()`, `cuda_available()`, `hmac_verify()`)
- [x] §5 Bench HMAC Rust vs Python avec test de réalité (`benchmarks/bench_hmac_rust_vs_python.py`)
- [x] §6 `admin/admin` default → mot de passe random sur disque mode 0600 ou `VRM_DEFAULT_ADMIN_PASS`, refus en prod (`core/auth_strong.py`)

Session 2026-05-11 (suite):

- [x] §0 Doc piège "0 streams" en B-2 warmup → header explicite ajouté dans `core/expert_pinning.py` (section "Metrics gotcha")
- [x] §5 PyO3 maturin wheel CI → ajout `build-sdist` + job `publish` PyPI OIDC déclenché sur tag dans `.github/workflows/build-rust.yml` ; clippy assoupli pour les warnings PyO3 macro-generated
- [x] §7 Grafana dashboards validés : 25 métriques utilisées toutes exposées dans `core/metrics.py` ; 16 alerting rules valides
- [x] §7 Healthcheck K8s : `/ready` retourne maintenant 503 si modèle non chargé (`VRM_READY_REQUIRE_MODEL=0` pour bypass) ; `monitoring/k8s-deployment-example.yaml` ajouté avec liveness/readiness/startup probes
- [x] §9 `_deprecated/` exclu du wheel via `[tool.setuptools.packages.find].exclude` + `MANIFEST.in` créé pour la sdist (validé : 0 package `_deprecated` détecté)
- [x] §11 Quickstart 5 min : `docs/QUICKSTART.md` existe déjà (110 lignes, complet)

Session 2026-05-11 (suite 2 — "fais au mieux") :

- [x] §2 **Backend auto-pick** : `recommend_backend()` dans `core/auto_detect.py` (heuristiques gguf → llamacpp, awq/gptq/fp8 → vllm, ollama:// → ollama, sinon huggingface ; fallback si non-installé) + 10 tests `tests/test_auto_detect_backend.py`
- [x] §2 **Hot-plug GPU/RAM** : `core/utils.py` détecte changement de `torch.cuda.device_count()` et invalide `_get_logical_mapping()` automatiquement + `invalidate_device_cache()` public
- [x] §6 **Audit log persistant** : `core/security/audit_log.py` (SQLite WAL, token hashé sha256[:16] jamais en clair, 3 index, thread-safe, never-raise) + before/after_request hooks dans `core/security/__init__.py` + 6 tests `tests/test_security_audit_log.py`
- [x] §7 **Alerting rules complémentaires** : 6 nouvelles règles dans `monitoring/alerting_rules.yml` (KV cache pressure/borrowing, batcher queue depth, circuit breaker open/flapping) → 22 rules / 9 groupes
- [x] §7 **Circuit breaker instrumenté** : `core/metrics.py` expose `CIRCUIT_BREAKER_STATE` (gauge 0/1/2) + `CIRCUIT_BREAKER_TRIPS` (counter), `core/api/circuit_breaker.py` publie l'état à chaque transition
- [x] §9 **README compatibility matrix** : ajout d'une matrice 16 lignes (backends + features) × 7 colonnes (OS + accelerators) dans `README.md`
- [x] **Bug fix critique** : `core/health.py` variable shadowing `result` → `nvml_result` (le NVML handle dict écrasait le dict de retour, surfaced via `pynvml` désormais installé)

**Suite tests : 1090 passed, 0 failed, 69 skipped** (gain +16 nouveaux tests, +1 bug pré-existant corrigé).

---

## 0. Court terme — finir V6.E proprement
- [ ] **Phase B-3** : préfetch prédictif des cold experts (router-history-based) pour overlap PCIe ↔ compute en `stream_every`. Cible : combler les 56-69% de gap mesurés vs B-2 sans payer le miroir 24.8 GB. *(L)*
- [ ] **Cohorte 5070 Ti FP8 KV** : trouver le ceiling ctx (`VRM_BENCH_MAX_MODEL_LEN=12288/16384`) avec B-2 warmup. *(S)*
- [x] ~~**Documenter le piège "0 streams" en B-2 warmup**~~ *(S)* — fait : header "Metrics gotcha" dans `core/expert_pinning.py`
- [ ] **Push de la branche `feat/v6-lending-cooperative`** + PR vers `main` avec résumé honnête (B-1 vs B-2, limitations, calibration). *(M)*

---

## 1. Stabiliser le cœur (toujours pertinent)
- [ ] Test e2e multi-GPU réel pour `core/transfer_manager.py` (P2P + CPU-staged fallback). *(L)*
- [ ] Plan de secours OOM dans `InferencePipeline` pour modèles répartis (eviction → CPU offload graceful). *(M)*
- [ ] Race condition pré-existante dans `test_chaos_concurrency.py::test_pipeline_concurrent_load` — désélectionnée en CI, à fixer ou retirer. *(M)*
- [x] ~~Schema versioning `core/persistence.py` (SQLite).~~ *(S)* — fait, v2 avec `schema_version` table + `created_at`
- [ ] Distribuer le circuit breaker (actuellement local par process, casse en gunicorn multi-worker). *(M)*

---

## 2. Auto-détection (axe transverse, important pour onboarding)
- [ ] **Détection topologie GPU au démarrage** : NVLink, P2P matrix, NUMA, PCIe gen/lanes, vendor mix → exposer dans `/api/system` et écrire un `topology.json` à côté de `config.yaml`. *(M)*
- [x] ~~**Backend auto-pick**~~ *(M)* — fait : `recommend_backend()` dans `core/auto_detect.py` (10 tests)
- [x] ~~**Quantization auto-pick** selon GPU~~ *(S)* — fait via `core/auto_detect.recommend_quantization()` + `VRM_QUANTIZATION=auto` câblé dans HuggingFaceBackend
- [ ] **Auto-tune `cpu_offload_gb`** en fonction de `model_size + max_model_len + free_vram`. *(M)*
- [x] ~~**Hot-plug GPU/RAM**~~ *(M)* — fait : `invalidate_device_cache()` + détection auto via `_LOGICAL_MAPPING_DEVICE_COUNT` dans `core/utils.py`
- [ ] **Détection ReBAR** dans `experimental/cross_vendor_bridge.py` : déjà détecté, pas exploité — câbler à `transfer_manager` Strategy 1.5. *(L)*
- [x] ~~**VM Proxmox détection auto**~~ *(S)* — fait via `core/auto_detect.detect_virtualization()` + `should_disable_p2p()` câblé dans `TransferManager`

---

## 3. Interfaces LLM utilisateur (gap majeur pour adoption)
- [ ] **CLI unifiée** : `vramancer serve <model>` / `vramancer chat` / `vramancer bench` / `vramancer gpu-list`. Aujourd'hui éparpillé entre `vramancer/main.py`, scripts ad-hoc et bench scripts. *(L)*
- [ ] **OpenAI-compat client SDK** : juste un wrapper `from vramancer import Client` autour de la route Flask, avec retries + streaming. *(M)*
- [ ] **Chat REPL terminal** propre (Rich/Textual) avec historique persistent, sliders température/top-p, switch de modèle live. *(M)*
- [ ] **Dashboard web** : retirer le GPU data hardcodé dans les templates, brancher sur `/api/gpu` réel. *(M)*
- [ ] **Model browser** : intégration HF Hub avec filtre par taille / quant supportée par les GPU détectés (lien avec axe Auto-détection). *(M)*
- [ ] **Mode "deux modèles"** : draft + verifier en speculative decoding via UI cocher 1 modèle = activation auto du draft compagnon (Qwen→0.5B, Llama→1B). *(M)*
- [ ] **Endpoints OpenAI manquants** : `/v1/embeddings`, `/v1/audio/*` (placeholder), function calling complet. *(L)*
- [ ] **Multi-tenant simple** : namespacing par token API, quotas par tenant. *(L)*

---

## 4. Swarm & Cluster (le plus prometteur, le moins fini)
- [ ] **Câbler `swarm_ledger.py` à l'orchestrateur** : aujourd'hui SQLite ledger fonctionnel mais orphelin. Routing des requêtes vers contributeurs réels, payout virtuel/monétaire. *(XL)*
- [ ] **Supervision API : retirer les NODES hardcodés** (`raspberrypi`, `jetson`, `workstation` fictifs dans `core/network/supervision_api.py`). Brancher sur `cluster_discovery.py` réel. *(M)*
- [ ] **Auto-join cluster** : un nouveau worker fait `vramancer join --leader <ip>` ou découverte mDNS auto, négocie ses capabilities, reçoit des layers. *(L)*
- [ ] **Layer sharding cross-node** : aujourd'hui la PP cross-node passe par VTP mais pas de scheduler global. Étendre `BlockOrchestrator` au multi-node. *(XL)*
- [ ] **Failover live** : si un node tombe, redistribuer ses layers sur les survivants sans drop la requête en cours. État interne dans `gpu_fault_tolerance.py` à étendre cluster-wide. *(XL)*
- [ ] **WebGPU node** (`webgpu_node.py`, 800 LOC) : task dispatcher incomplet, mock auto-complete. À finir si on veut vraiment des navigateurs comme workers. *(L)*
- [ ] **AITP/RAID stress test cross-node réel** : 3+ machines, mesure recovery FEC sous packet loss simulé. *(M)*
- [ ] **Bench cross-node WAN** : on a `bench_wan_4g.py` (memo) — refaire avec setup actuel (LAN gigabit, 10G, WAN 4G/5G). *(M)*

---

## 5. Rust core / Tokio (potentiel énorme, peu exploité)
État : `rust_core/` compile (1.6 MB .so), CUDA FFI réel via `libloading`, HMAC 100x plus rapide que Python, triple-buffering DtoD non câblé.

- [ ] **Câbler `direct_vram_copy()` Rust → `transfer_manager.py` Strategy 1.5** : aujourd'hui le triple-buffer Rust existe mais n'est pas exposé via PyO3 vers la couche Python (memo dit "non câblé"). Vérifier l'état réel et finir le binding. *(L)*
- [ ] **Tokio runtime pour `network_raid.py`** : actuellement ThreadPoolExecutor. Réécrire le shard dispatch en Rust async/Tokio → gain attendu sur les très grosses sharded transfers. *(L)*
- [ ] **Rust event loop pour `cluster_discovery.py`** : Bully election + heartbeat = candidat parfait pour Tokio (low-latency, beaucoup de timers concurrents). *(L)*
- [ ] **Rust kernel pour `kv_quantizer.py`** : 56 compress/token sur 7B, GIL lock cher. CUDA via cudarc + cargo feature. *(XL)*
- [x] ~~**Fallback Python gracieux quand `rust_core` absent**~~ *(S)* — fait : `core/rust_bridge.py` façade safe, `cuda_available()` exposé, `Option<Library>` côté Rust
- [ ] **Linux-only contrainte** : `libcuda.so.1` seulement. ~~Ajouter Windows (`nvcuda.dll`)~~ partiellement fait (Rust try `nvcuda.dll` sur Windows) + macOS (Metal via `metal-rs`?). *(L)*
- [x] ~~**Bench Rust HMAC vs Python**~~ *(S)* — fait : `benchmarks/bench_hmac_rust_vs_python.py` avec warning si <3x speedup
- [ ] **PyO3 + maturin wheel CI** : déjà existante, vérifier qu'elle publie vraiment des wheels usables sur PyPI. *(S)*

---

## 6. Sécurité durcie (bloquant pour prod B2B)
- [x] ~~**Retirer `admin/admin` default en dev**~~ *(S)* — fait : random password sur disque mode 0600 ou `VRM_DEFAULT_ADMIN_PASS`, refus explicite en prod
- [ ] **MFA / SSO** : aujourd'hui `auth_strong.py` = JWT + PBKDF2 seul. Au minimum SSO OIDC pour entreprises. *(L)*
- [x] ~~**Audit log persistent**~~ *(M)* — fait : `core/security/audit_log.py` (SQLite WAL, token hashé, hooks Flask) + 6 tests
- [ ] **Rate limiting distribué** (Redis backend) au lieu du local per-IP actuel. *(M)*
- [ ] **Tests sécurité** : `verify_request()`, `enforce_startup_checks()`, prompt injection detection. *(M)*
- [ ] **Secrets management** : aujourd'hui ENV vars. Intégration Vault / AWS Secrets / SOPS optionnelle. *(M)*

---

## 7. Observabilité production
- [ ] **OpenTelemetry traces complètes** : `core/tracing.py` est OK mais pas instrumenté partout (pipeline → backend → transfer). *(M)*
- [x] ~~**Grafana dashboards prêts à l'emploi**~~ *(S)* — validé : 25 métriques toutes exposées, 16 alerting rules valides, provisioning OK
- [x] ~~**Alerting rules**~~ *(M)* — fait : 22 rules / 9 groupes (KV cache pressure, borrowing, batcher queue, circuit breaker open/flapping)
- [x] ~~**Healthcheck Kubernetes-ready**~~ *(S)* — fait : `/ready` retourne 503 si modèle absent (override `VRM_READY_REQUIRE_MODEL=0`), manifest `monitoring/k8s-deployment-example.yaml` avec liveness/readiness/startupProbe

---

## 8. Performances (axe permanent)
- [ ] **Bare-metal vs VM Proxmox** : mesurer overhead VFIO. Si <5%, tant mieux ; si >15%, documenter clairement. *(M)*
- [ ] **CUDA Graph multi-GPU** : actuellement single-GPU only (`cuda_graph_decode.py`). *(XL)*
- [ ] **Transfer overlap PP** : prefetch couche N+1 pendant compute couche N. Aujourd'hui PP sériel. *(L)*
- [ ] **Sync points pipeline parallel** : profiler avec Nsight, identifier les barrières inutiles. *(L)*
- [ ] **Triton fused sampling** : `triton_sampling.py` fallback PyTorch toujours utilisé en pratique, le triton_full path est peu emprunté. Diagnostiquer pourquoi (cf. `VRM_DEBUG_SAMPLING=1`). *(M)*
- [ ] **Benchmark continuous batcher concurrent** : 10/50/100 utilisateurs concurrents, mesurer p50/p95/p99 latency. *(L)*

---

## 9. Qualité code & dette technique
- [x] ~~**Module central `core/env_flags.py`**~~ *(M)* — fait : 140 flags registry + dump_flags/dump_active/unknown_env_flags + CLI
- [ ] **Supprimer / re-implémenter le dead code** :
  - [ ] `core/backends_webgpu.py` POC (déjà marqué "Production Ready = FAUX") *(S delete / XL implement)*
  - [x] ~~`core/telemetry.py`~~ — **PAS orphelin** : 3 consommateurs réels (`vramancer/cli/telemetry_cli.py`, `core/network/supervision_api.py`, tests). Audit corrigé.
  - [ ] `core/orchestrator/heterogeneous_manager.py` GPU scoring hardcodé string-match → mesure réelle. *(M)*
  - [ ] `core/orchestrator/block_orchestrator.py` benchmarks single-block trompeurs *(M)*
- [ ] **Docstrings honnêtes** : déjà commencé (RemoteExecutor "zero-copy" → corrigé, `cross_vendor_bridge.py` DMA-BUF → corrigé). Étendre à tous les modules grade C. *(M)*
- [x] ~~**Renommer `software_cxl.cpp` → `file_offload.cpp`**~~ *(S)* — fait, header explicite "plain file I/O, NOT CXL hardware"
- [x] ~~**Archiver `_deprecated/`** dans le packaging~~ *(S)* — fait : `pyproject.toml` `[tool.setuptools.packages.find].exclude` + `MANIFEST.in` `prune _deprecated` (validé : 0 package détecté)
- [x] ~~**Matrice de compat backend/OS/GPU dans le README**~~ *(M)* — fait : tableau 16×7 (backends + features × OS + accelerators) dans `README.md` ; distinction production-ready vs experimental reste à affiner.

---

## 10. Packaging & distribution
- [ ] **`pip install vramancer && vramancer serve <model>` en 5 min** : c'est l'objectif onboarding. Aujourd'hui il faut config + launchers + vérifs. *(L)*
- [ ] **Docker multi-stage officiel** : `Dockerfile` existe, vérifier qu'il marche pour cuda+rocm en image unique ou splitter. *(M)*
- [ ] **Wheels PyPI** : auto-publish sur tag `v*`. *(M)*
- [ ] **Conda-forge package** (bonus). *(M)*
- [ ] **Homebrew tap macOS** (CPU/MPS). *(M)*
- [ ] **Helm chart Kubernetes** pour déploiement cluster. *(L)*

---

## 11. Documentation utilisateur (axe sous-investi)
- [x] ~~**Quickstart 5 min**~~ *(S)* — `docs/QUICKSTART.md` existe (110 lignes, install + verify + serve + curl OpenAI-compat)
- [ ] **Use cases tutorials** :
  - "Faire tourner Llama-70B sur 2× RTX 3090"
  - "Mix RTX 5090 + 3090 sans perdre 30% des perfs"
  - "Self-hosted ChatGPT pour mon équipe"
  - "Edge deployment Jetson + dashboard remote"
- [ ] **API reference auto-générée** (Sphinx ou MkDocs avec mkdocstrings). *(M)*
- [ ] **Architecture decision records** : pourquoi PP par défaut vs TP, pourquoi pas torchrun, etc. *(M)*
- [ ] **Migration guide vLLM → VRAMancer** pour adoption. *(M)*

---

## 12. Validation externe (bloquant pour crédibilité)
- [ ] **Bench rigoureux vs vLLM/TGI/llama.cpp/Ollama** sur même hardware, mêmes prompts, mêmes modèles. Publier brut sans cherry-picking. *(L)*
- [ ] **2-3 early adopters production** (open-source projects, étudiants/labs, indie devs). *(L)*
- [ ] **Stress test multi-user 24h** : continuous batcher sous charge réelle 50+ users concurrents. *(L)*
- [ ] **Soumission HF Spaces / Replicate** : un demo public. *(M)*
- [ ] **Article technique honnête** : ce qui marche, ce qui ne marche pas, les chiffres. *(M)*

---

## Priorités stratégiques (proposition)

**Sprint immédiat (1-2 semaines)** : 0 + 3 (CLI unifiée + chat REPL) + 11 (quickstart) → rendre le projet *essayable* en 5 min.

**Sprint moyen (1-2 mois)** :
- 2 (auto-détection)
- 5 (Tokio/Rust câblage des chemins existants — ce qui rend le projet techniquement différenciant)
- 9 (nettoyage dead code)

**Sprint long (3-6 mois)** :
- 4 (swarm — c'est *the killer feature* si fini)
- 12 (validation externe — sans early adopters le projet n'existe pas)
- 6 (sécurité durcie — bloquant B2B)

**Toujours en arrière-plan** : 1, 7, 8, 10.

---

## Notes axe par axe (réponse à ta question)

- **Swarm** → axe le plus différenciant et le moins fini. `cluster_discovery.py` réel mais `swarm_ledger.py` orphelin et `supervision_api.py` plein de nodes fictifs. Ce serait *the moat*, mais c'est XL.
- **Interfaces LLM** → gros gap onboarding. CLI unifiée + chat REPL + dashboard nettoyé = quick wins moyenne taille.
- **Auto-détection** → indispensable pour le "pip install && ça marche". Topologie + backend pick + quant pick = chemin clair.
- **Tokio/Rust** → bon levier perf mais beaucoup déjà "présent et non câblé". Plus de gain à câbler le triple-buffer existant qu'à réécrire en Rust ce qui marche déjà en Python (HMAC, network_raid).
