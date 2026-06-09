# Résultats d'exécution — Plan V2 Architecte

**Branche :** `chore/sonnet-plan-v2`  
**Baseline :** 1014 passed, 1 failed (pré-existant), 39 skipped  
**Final :** **1070 passed**, 1 failed (même pré-existant), 39 skipped  
**Nouveaux tests ajoutés :** +56

---

## P0 — Élimination des effets de bord à l'import

### V0.1 — Guard ClusterDiscovery (`core/api/registry.py`)
- **Commit :** `01c97da`
- `PipelineRegistry.__init__` : ClusterDiscovery auto-start conditionnel à `VRM_CLUSTER_AUTO_DISCOVER=1`
- Validation : `python -c "from core.api.registry import PipelineRegistry; r = PipelineRegistry(); assert r.discovery is None"` → OK

### V0.2 — Guard VTP server (`core/production_api.py`)
- **Commit :** `36f4bbf`
- `start_vtp_server()` conditionnel à `VRM_FEATURE_AITP=1`
- Validation : `cross_node` non importé sans flag → OK

### V0.3 — Nouveaux flags env (`core/env_flags.py`)
- **Commit :** `adeb401`
- Ajout `CLUSTER_AUTO_DISCOVER` (bool, VRM_CLUSTER_AUTO_DISCOVER)
- Ajout `FEATURE_AITP` (bool, VRM_FEATURE_AITP)

### V0.4 — Tests side-effects-free (`tests/test_no_import_side_effects.py`)
- **Commit :** `1d56f6b`
- **3 tests** : registry sans discovery, production_api sans VTP, import pipeline sans threads GPU
- Résultat : 3 passed ✅

---

## P1 — Migration env_flags dans inference_pipeline

### V1.1 — Import facade (`core/inference_pipeline.py`)
- **Commit :** `0d80688`
- Ajout `from core.env_flags import flags as _flags` avec fallback `_flags = None`

### V1.2 — Migration 21 `os.environ.get` → `_flags.X`
- **Commit :** `a63db64`
- 21 remplacements en 3 passes (L128-L523, L793-L848, L1100-L1519)
- Flags migrés : REBALANCE_INTERVAL, PARALLEL_MODE, VRAM_LENDING, DRAFT_MODEL, SPEC_GAMMA, SPEC_ADAPTIVE, GENERATE_TIMEOUT, DISABLE_TURBO, CUDA_GRAPH, CUDA_GRAPH_CACHE, CUDA_GRAPH_WARMUP, FORCE_MULTI_GPU, QUANTIZATION, KV_COMPRESSION, KV_COMPRESSION_BITS, KV_CACHE_RESIDUAL, MAX_BATCH_SIZE, LEND_RATIO, RECLAIM_THRESHOLD, LENDING_INTERVAL (+1 dupliqué)
- Validation : `test_pipeline.py` → 37 passed ✅

### V1.3 — Test suite env_flags (`tests/test_env_flags.py`)
- **Commit :** `1308da0`
- **22 tests** : bool defaults, int defaults, float defaults, str defaults, live mutation
- Résultat : 22 passed ✅

---

## P2 — Tests sécurité

### V2.1 — verify_request tests (`tests/test_verify_request.py`)
- **Commit :** `aec3133`
- **14 tests** : public paths, production 401, token valid, token invalid, relax_security bypass, bypass_ha
- Résultat : 14 passed ✅

### V2.2 — startup_checks tests (`tests/test_startup_checks.py`)
- **Commit :** `c544a2c`
- **7 tests** : non-prod no-error, prod requires token, prod requires secret, rejects MINIMAL_TEST/relax_security/bypass_ha, prod all correct
- Résultat : 7 passed ✅

---

## P3 — Honnêteté du code

### V3.1 — Docstring RemoteExecutor (`core/block_router.py`)
- **Commit :** `7dc02b4`
- Docstring corrigée : "NOT zero-copy — safetensors → TCP socket round-trip"

### V3.2 — Mode banner dans pipeline.load() (`core/inference_pipeline.py`)
- **Commit :** `7dc02b4`
- Log : `Loading model: {name} (backend={backend}) [quant=NF4 | parallel=PP | cuda_graph=ON]`
- Format dynamique depuis `_flags`

---

## P4 — Tests LlamaServerBackend

### V4.1 — Tests unitaires (`tests/test_llama_server_backend.py`)
- **Commit :** `8a01ab1`
- **9 tests** : `_platform_key()` retourne string valide, darwin-arm, darwin-x86, windows, `_ASSET_MAP` tag placeholder, SERVER_PORT default, BINARY_DIR sous home, init raises sans binaire
- Résultat : 9 passed ✅

---

## P5 — Documentation

### V5.1 — COMPATIBILITY.md (`docs/COMPATIBILITY.md`)
- **Commit :** `aa481ba`
- Matrice backend × quantization (BF16/NF4/INT8/NVFP4/GGUF)
- Matrice backend × OS (Linux/macOS/Windows)
- Matrice accélérateur × backend (CUDA/ROCm/MPS/XPU/CPU)
- Tableau multi-GPU strategies
- Tableau KV compression
- VRAM Lending Pool conditions
- Section limitations connues

---

## P6 — Couverture imports

### V6.1 — `core.production_api` dans CORE_MODULES (`tests/test_imports.py`)
- **Commit :** `aa481ba`
- `"core.production_api"` ajouté à la liste (était absent)
- `test_imports.py` → 48 passed ✅

---

## P7 — Validation finale

| Métrique | Baseline | Final | Delta |
|---------|---------|-------|-------|
| Tests passed | 1014 | **1070** | **+56** |
| Tests failed | 1 (pré-existant) | 1 (même) | 0 |
| Tests skipped | 39 | 39 | 0 |
| Commits ajoutés | — | **10** | — |

### Commits sur `chore/sonnet-plan-v2`

```
aa481ba [V5.1+V6.1] add COMPATIBILITY.md, add core.production_api to test_imports
8a01ab1 [V4.1] add test_llama_server_backend.py (9 tests)
7dc02b4 [V3.1+V3.2] fix RemoteExecutor docstring, add mode banner to pipeline.load()
c544a2c [V2.2] add test_startup_checks.py (7 tests)
aec3133 [V2.1] add test_verify_request.py (14 tests)
1308da0 [V1.3] add test_env_flags.py (22 tests)
a63db64 [V1.2] migrate 21 os.environ.get to env_flags facade in inference_pipeline
0d80688 [V1.1] import env_flags facade in inference_pipeline
1d56f6b [V0.4] add side-effects-free import tests (3 tests)
adeb401 [V0.3] add CLUSTER_AUTO_DISCOVER and FEATURE_AITP to env_flags
36f4bbf [V0.2] guard VTP server start behind VRM_FEATURE_AITP
01c97da [V0.1] guard ClusterDiscovery auto-start behind VRM_CLUSTER_AUTO_DISCOVER
```

### Fichiers modifiés ou créés

| Fichier | Action | Tâche |
|---------|--------|-------|
| `core/api/registry.py` | Modifié | V0.1 |
| `core/production_api.py` | Modifié | V0.2 |
| `core/env_flags.py` | Modifié | V0.3 |
| `core/inference_pipeline.py` | Modifié | V1.1, V1.2, V3.2 |
| `core/block_router.py` | Modifié | V3.1 |
| `docs/COMPATIBILITY.md` | Créé | V5.1 |
| `tests/test_no_import_side_effects.py` | Créé | V0.4 |
| `tests/test_env_flags.py` | Créé | V1.3 |
| `tests/test_verify_request.py` | Créé | V2.1 |
| `tests/test_startup_checks.py` | Créé | V2.2 |
| `tests/test_llama_server_backend.py` | Créé | V4.1 |
| `tests/test_imports.py` | Modifié | V6.1 |
