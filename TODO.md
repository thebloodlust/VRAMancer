# VRAMancer Roadmap

## 1. Stabiliser le cœur du projet
- [ ] Valider `InferencePipeline.load() -> generate() -> shutdown()` sur de vrais backends : HuggingFace, vLLM, llama.cpp. *(L)*
- [ ] Ajouter un test end-to-end multi-GPU réel pour `core/transfer_manager.py` avec P2P et CPU-staged fallback. *(L)*
- [x] Séparer clairement les chemins de production des modes test (`VRM_MINIMAL_TEST`, `VRM_TEST_MODE` dans conftest.py + CI).
- [ ] Ajouter un flag explicite `VRM_FEATURE_AITP=1` pour activer `core/network/llm_transport.py`. *(S)*
- [ ] Retirer les effets de bord réseau/imports lourds lors de l'import de `core/api/production_api.py`. *(M)*
- [ ] Implémenter un plan de secours OOM dans `core/inference_pipeline.py` pour les modèles répartis. *(M)*

## 2. Sécurité et API
- [x] Vérifier que `VRM_PRODUCTION=1` bloque tous les bypass de sécurité et active `startup_checks` — implémenté dans `core/security/__init__.py` et `startup_checks.py`.
- [ ] Ajouter des tests de sécurité pour `core/security.verify_request()` et `startup_checks.enforce_startup_checks()`. *(M)*
- [ ] S'assurer que `/api/pipeline/status`, `/api/queue/status` et `/api/gpu` sont exposés et stables. *(M)*
- [ ] Documenter une matrice de compatibilité backend / OS / GPU / quantization. *(M)*

## 3. WebGPU, WebNPU et Edge Compute
- [ ] Stabiliser le protocole binaire WebSocket (`core/webgpu_backend.py` ↔ `dashboard/worker/`). *(M)*
- [ ] Ajouter des tests unitaires pour le backend WebGPU (matmul round-trip, reconnection). *(M)*
- [ ] Documenter le déploiement WebNPU (WebNN) : devices supportés, fallback WebGPU. *(S)*
- [ ] Benchmarker la latence WAN (4G/5G) pour les workers Edge via `benchmarks/bench_wan_4g.py`. *(S)*

## 4. Documentation et packaging
- [ ] Distinguer dans le README les modules `production-ready` et `experimental`. *(M)*
- [ ] Documenter les dépendances optionnelles : `rust_core`, `csrc`, `pyverbs`, `nvidia_peermem`, `llama-cpp-python`, `bitsandbytes`. *(S)*
- [ ] Archiver ou exclure `_deprecated/` du packaging et du runtime principal. *(S)*
- [ ] Ajouter un guide d'installation clair pour Linux/CUDA, macOS/MPS, Windows/CUDA, CPU-only. *(M)*

## 5. Nettoyage et refactorisation
- [ ] Créer un module central `core/env_flags.py` pour regrouper les ~120 `VRM_*` flags. *(M)*
- [ ] Isoler `core/orchestrator/` : clarifier `PlacementEngine` vs `BlockOrchestrator`. *(M)*
- [ ] Réduire l'emprise de `supervision_api.py` et documenter son périmètre réel. *(S)*
- [ ] Nettoyer les imports de modules obsolètes dans `core/backends.py` et `core/production_api.py`. *(S)*

## 6. Réseau et AITP
- [ ] Documenter l'état réel d'`AITP` : matériel requis, dépendances, fallback, utilisations prévues. *(M)*
- [ ] Ajouter un test d'intégration réseau pour `core/network/llm_transport.py`, stub et réel. *(M)*
- [ ] Marquer `core/network/network_transport.py` comme optionnel et ne pas le rendre obligatoire au démarrage. *(S)*

## 7. Rust et Tokio
- [x] CI Rust en place (`build-rust.yml` : lint clippy, cargo test, maturin wheels multi-OS).
- [ ] Documenter `rust_core/` comme extension optionnelle, avec `cargo build --release --features cuda`. *(S)*
- [ ] Fournir un fallback Python propre quand `rust_core` est absent. *(S)*
- [ ] Vérifier que `rust_core` n'est pas requis pour les fonctionnalités de base. *(S)*

## 8. Tests et CI
- [x] Jobs CI séparés : `ci.yml` (stub/minimal multi-OS), `gpu-tests.yml`, `build-rust.yml`.
- [x] `tests/test_imports.py` existe — à étendre pour couvrir tous les modules core.
- [ ] Ajouter un benchmark smoke GPU pour vérifier la compatibilité de l'environnement. *(S)*
- [ ] Ajouter un test de configuration pour `torch`, `transformers`, `bitsandbytes`, `pyverbs`. *(S)*

## 9. Observabilité et maintenance
- [x] `reset_metrics()` existe dans `core/metrics.py` et est appelé dans `InferencePipeline.shutdown()`.
- [x] `reload_config()` existe dans `core/config.py`.
- [ ] Documenter la version de schéma SQLite dans `core/persistence.py`. *(S)*
- [ ] Logguer clairement les modes `stub`, `production`, `experimental` au démarrage. *(S)*

## Notes de priorité
1. **Stabiliser le cœur** de l'inférence et les backends réels.
2. **WebGPU/WebNPU** — axe actif, résultats prometteurs (67 tok/s decode S25 Ultra).
3. **Centraliser les flags** — ~120 `VRM_*` éparpillés → `core/env_flags.py`.
4. Renforcer la sécurité et l'API production.
5. Définir `AITP`/`rust_core` comme extensions optionnelles.

## Tailles estimées
- **(S)** < 1h — flag, doc, test simple
- **(M)** 1–4h — refactoring, module, test d'intégration
- **(L)** > 4h — validation end-to-end, multi-GPU, architecture
