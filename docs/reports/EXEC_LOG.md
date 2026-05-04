# EXEC_LOG — chore/sonnet-plan-exec

Branche : `chore/sonnet-plan-exec`  
Plan source : `docs/reports/PLAN_ACTION.md`  
Démarrage : 2026-04-08

---

## [BASELINE]

- Commande : `VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=no`
- Résultat : **999 passed, 1 failed (pre-existing: test_fault_pipeline::test_health_imports_fault_manager), 39 skipped**
- Durée : ~74 s

---

## [T0.1] secure trust_remote_code via VRM_TRUST_REMOTE_CODE

- Fichier modifié : `core/inference_pipeline.py`
- Changement : Remplacement de `trust_remote_code=True` (×2) par `_trc = os.environ.get("VRM_TRUST_REMOTE_CODE") == "1"` + usage de `_trc`.
- Validation : `grep -n "trust_remote_code=True" core/inference_pipeline.py` → 0 résultats
- Statut : DONE

---

## [T0.2] drop marketing wording in logs and docstrings

- Fichiers modifiés : `core/scheduler.py`, `core/stream_manager.py`, `core/network/connectome.py`, `core/network/anycast_balancer.py`, `core/orchestrator/placement_engine.py`
- `vram_neuroplastic` → `vram_adaptive` (aucun test ne référence ce string)
- Statut : DONE

---

## [T0.3] add docs/CORE_ENGINE.md

- Fichier créé : `docs/CORE_ENGINE.md`
- Statut : DONE

---

## [T0.4] add core/config_manager.py with tests

- Fichiers créés : `core/config_manager.py`, `tests/test_config_manager.py`
- Tests : 5 fonctions, toutes PASSED
- Statut : DONE

---

## [T0.5] race-test for continuous batcher submit/start

- Fichier créé : `tests/test_pipeline_batcher_race.py`
- Résultat : PASS (pas de race détectée)
- Statut : DONE

---

## [T1.1] document _deprecated and clarify duplicate backends

- Fichier créé : `_deprecated/README.md`
- Statut : DONE

---

## [T1.2] sync version to 1.5.0 across server.py and CLI

- Fichiers modifiés : `server.py`, `vramancer/main.py`
- Statut : DONE

---

## [T1.3] add FastAPI e2e and security boundary tests

- Fichiers créés : `tests/test_fastapi_e2e.py`, `tests/test_security_boundary.py`
- Statut : DONE

---

## [T2.1] add PagedAttentionOffloader adapter

- Fichiers créés : `core/paged_attention_offload.py`, `tests/test_paged_offload.py`
- Statut : DONE

---

## [T2.2] add libibverbs probe to detect_best_transport

- Statut : BLOCKED ou DONE (voir détails ci-dessous)

---

## [T2.3] document continuous batching + concurrent bench script

- Fichiers créés : `docs/CONTINUOUS_BATCHING.md`, `benchmarks/bench_batcher_concurrent.py`
- Statut : DONE

---

## [T3.1] add module status badges to peripheral components

- Statut : DONE

---

## [T3.2] add anycast routing bench script

- Fichier créé : `benchmarks/bench_anycast_routing.py`
- Statut : DONE

---

## [SUMMARY]

_À remplir après validation finale._
