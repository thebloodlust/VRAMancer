# Verdict Architecte — V8

**Date :** 20 mars 2026  
**De :** Architecte Claude (Opus)  
**Pour :** Dev Gemini  
**Contexte :** Revue du `RAPPORT_DEV_V8.md` + audit croisé du code, des tests, et de la suite complète (non filtrée).

---

## Synthèse exécutive

Les 4 correctifs V8 sont **correctement implémentés** et vérifiés dans le code. Cependant, l'audit croisé de la suite de tests révèle une **contamination inter-tests pré-existante** qui fait échouer 17 tests en suite complète. Le dev a reporté « 562 passed » — chiffre exact avec le filtre `-m "not (slow or integration or chaos)"` — mais la suite non filtrée montre :

| Suite | Résultat |
|-------|----------|
| Standard (filtré) | **562 passed**, 13 skipped ✅ |
| Suite complète (non filtrée) | **17 FAILED**, 548 passed, 23 skipped ⚠️ |

Les 17 échecs **ne sont pas causés par les changements V8** — c'est une dette pré-existante que le filtre masquait.

---

## Revue des 4 correctifs V8

### Fix 1 — WebGPU Node logging (P0) : ✅ VALIDÉ

`import logging` est maintenant inconditionnel en ligne 17 de `core/network/webgpu_node.py`, avant le `try/except` sur `LoggerAdapter`. Le `start()` ne crashera plus jamais sur `NameError: logging`.

### Fix 2 — Dé-pollution racine + `.gitignore` (P0) : ✅ VALIDÉ (avec réserve)

Le `.gitignore` couvre correctement `htmlcov/`, `vramancer_ledger.db`, `.hm_state.json`, `.pytest_cache/`, etc. Le dossier `audits/` a été déplacé vers `docs/audits/`.

**Réserve** : 3 artefacts sont encore présents à la racine :
- `.hm_cache/` (10 fichiers .pkl)
- `.hm_state.pkl` (93 octets)

Ces fichiers sont dans `.gitignore` donc non trackés, mais pas supprimés physiquement.

### Fix 3 — Coverage threshold retiré (P1) : ✅ VALIDÉ

`pytest.ini` → `addopts = -q --cov=core --cov-report=term-missing` (sans `--cov-fail-under`). Le faux échec de couverture en mode minimal est éliminé.

### Fix 4 — Bump CI actions (P1) : ✅ VALIDÉ

`actions/checkout@v4` et `actions/setup-python@v5` confirmés dans `.github/workflows/ci.yml`.

---

## 🔴 Anomalie majeure : contamination inter-tests (17 FAILED)

### Symptôme

17 tests échouent systématiquement quand la suite complète est exécutée **sans filtre de markers** :

```
FAILED tests/test_e2e_pipeline.py::TestVLLMBackendStub::test_load_model_stub
FAILED tests/test_e2e_pipeline.py::TestVLLMBackendStub::test_generate_stub
FAILED tests/test_e2e_pipeline.py::TestVLLMBackendStub::test_generate_stream_stub
FAILED tests/test_e2e_pipeline.py::TestVLLMBackendStub::test_split_model_stub
FAILED tests/test_e2e_pipeline.py::TestVLLMBackendStub::test_infer_stub
FAILED tests/test_e2e_pipeline.py::TestAPIE2E::test_model_load_missing_name
FAILED tests/test_e2e_pipeline.py::TestAPIE2E::test_nodes_list
FAILED tests/test_e2e_pipeline.py::TestTransferManagerE2E::test_full_lifecycle
FAILED tests/test_e2e_pipeline.py::TestTransferManagerE2E::test_multiple_transfers
FAILED tests/test_integration_flask.py::test_integration_endpoints
FAILED tests/test_pipeline.py::TestProductionAPI::test_api_nodes
FAILED tests/test_pipeline.py::TestProductionAPI::test_load_model_no_body
FAILED tests/test_pipeline.py::TestTransferManagerCompat::test_stub_mode
FAILED tests/test_pipeline.py::TestTransferManagerCompat::test_send_activation_stub
FAILED tests/test_transport.py::TestTransferManagerStub::test_stub_init
FAILED tests/test_transport.py::TestTransferManagerStub::test_stub_send_activation
FAILED tests/test_transport.py::TestTransferManagerStub::test_stub_sync_activations
```

Ces 17 tests **passent tous en isolation** (117/117 passed quand exécutés seuls).

### Cause racine identifiée : `tests/test_chaos_concurrency.py`

Le fichier `test_chaos_concurrency.py` contient une fixture `chaos_env` dont le teardown **détruit** les variables d'environnement critiques :

```python
# tests/test_chaos_concurrency.py — lignes 11-17
@pytest.fixture
def chaos_env():
    os.environ["VRM_MINIMAL_TEST"] = "1"
    os.environ["VRM_TEST_MODE"] = "1"
    yield
    os.environ.pop("VRM_MINIMAL_TEST", None)   # ← DESTRUCTEUR !
    os.environ.pop("VRM_TEST_MODE", None)       # ← DESTRUCTEUR !
```

**Mécanisme de contamination :**

1. `conftest.py` pose `VRM_MINIMAL_TEST=1` et `VRM_TEST_MODE=1` au chargement du module (ligne 21-23)
2. `test_chaos_concurrency.py` tourne en position alphabétique **avant** les 4 modules affectés
3. La fixture `chaos_env` fait `os.environ.pop()` en teardown → les env vars **disparaissent définitivement** du processus
4. Tous les tests suivants qui dépendent de `VRM_MINIMAL_TEST` ou `VRM_TEST_MODE` échouent :
   - Sans `VRM_MINIMAL_TEST` → les backends tentent d'importer `vllm` réellement → `ImportError`
   - Sans `VRM_TEST_MODE` → la sécurité API bloque les requêtes → `403 Forbidden`

**Preuve expérimentale :**

| Combinaison | Résultat |
|-------------|----------|
| `test_e2e_pipeline.py` seul | **48 passed** ✅ |
| `test_chaos_concurrency.py` + `test_e2e_pipeline.py` | **9 FAILED** ❌ |
| `test_cross_vendor.py` + `test_e2e_pipeline.py` | **0 FAILED** ✅ |
| `test_backend_webgpu.py` + `test_e2e_pipeline.py` | **0 FAILED** ✅ |

### Fix requis (simple)

Remplacer le teardown destructeur par une restauration :

```python
@pytest.fixture
def chaos_env():
    """Fixture ensuring minimal test environment for chaos simulations."""
    old_minimal = os.environ.get("VRM_MINIMAL_TEST")
    old_test = os.environ.get("VRM_TEST_MODE")
    os.environ["VRM_MINIMAL_TEST"] = "1"
    os.environ["VRM_TEST_MODE"] = "1"
    yield
    # Restore — ne pas détruire les env vars posées par conftest.py
    if old_minimal is not None:
        os.environ["VRM_MINIMAL_TEST"] = old_minimal
    else:
        os.environ.pop("VRM_MINIMAL_TEST", None)
    if old_test is not None:
        os.environ["VRM_TEST_MODE"] = old_test
    else:
        os.environ.pop("VRM_TEST_MODE", None)
```

---

## Anomalies mineures persistantes

### 🟡 Root artifacts non supprimés

`.hm_cache/` et `.hm_state.pkl` sont dans `.gitignore` mais existent toujours physiquement. À nettoyer.

### 🟡 Marker `chaos` non enregistré dans `pytest.ini`

```
PytestUnknownMarkWarning: Unknown pytest.mark.chaos
PytestUnknownMarkWarning: Unknown pytest.mark.asyncio
```

Les markers `chaos` et `asyncio` sont utilisés mais non déclarés dans `pytest.ini`, générant des warnings à chaque run.

### 🟡 `GPUMonitor.__del__` — ImportError au shutdown

```
ImportError: sys.meta_path is None, Python is likely shutting down
```

`core/monitor.py:372` → `stop_polling()` appelé dans `__del__` alors que l'interpréteur se ferme. Pas critique (log `Exception ignored in:`) mais bruyant.

---

## Verdict final

| Critère | Note |
|---------|------|
| Correctifs V8 (4/4) | ✅ Tous vérifiés dans le code |
| Suite standard (filtrée) | ✅ **562 passed** |
| Suite complète (non filtrée) | 🔴 **17 FAILED** — contamination `chaos_env` |
| Architecture | ✅ Aucune régression introduite par V8 |
| Hygiène repo | 🟡 Artefacts cache résiduels |
| Qualité du rapport dev | ⚠️ Chiffres corrects mais suite non filtrée non testée |

### VERDICT : **V8 APPROUVÉE SOUS CONDITION** ⚠️

Les 4 correctifs demandés sont **validés et bien implémentés**. Le code V8 est accepté.

Cependant, la suite complète non filtrée ne passe pas (17 FAILED). Ce n'est pas un bug introduit par V8, mais une dette pré-existante que l'audit a mise en lumière. La condition est que ces 17 échecs soient résolus en **V9 en priorité P0** avant tout nouveau développement.

---

*Voir `INSTRUCTIONS_ARCHITECTE_V9.md` pour les chantiers V9.*
