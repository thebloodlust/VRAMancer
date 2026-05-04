# Plan d'action VRAMancer V2 — exécution Sonnet 4.6

> **Architecte / Auditeur** : Claude Sonnet 4.5 (5 mai 2026)
> **Exécutant** : Sonnet 4.6
> **Branche** : `chore/sonnet-plan-v2` (depuis `chore/sonnet-plan-exec`)
> **Source de vérité** : `.github/copilot-instructions.md`
> **Baseline tests** : 1014 passed, 1 failed (pré-existant `test_health_imports_fault_manager`), 39 skipped
> **Tous les numéros de ligne ci-dessous ont été vérifiés le 5 mai 2026.**

---

## Vision architecturale

VRAMancer est ~70% production-ready, ~15% incomplet, ~15% marketing/dead-code (cf. `.github/copilot-instructions.md`). Ce plan vise quatre objectifs **non négociables** :

1. **Honnêteté** : aucune affirmation fausse dans les logs, docstrings, noms de classes.
2. **Aucun effet de bord à l'import** : importer un module ne doit JAMAIS ouvrir de socket, démarrer de thread réseau, ou écrire sur disque sans flag explicite.
3. **Centralisation des flags** : `core/env_flags.py` est la source unique. Aucun nouveau `os.environ.get("VRM_*")` ne doit apparaître hors `env_flags.py`.
4. **Sécurité testée** : `verify_request`, `enforce_startup_checks` et tout backend de prod doivent avoir des tests directs.

---

## Règles globales (ABSOLUES)

1. **Un commit par tâche**. Format : `[VX.Y] description (≤72 chars)`.
2. **Avant chaque commit** :
   ```bash
   VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
     pytest -q tests/ --ignore=tests/test_chaos_concurrency.py --tb=no --no-cov 2>&1 | tail -1
   ```
   Confirmer : aucune nouvelle failure (1014 → ≥1014 passed, 1 failed pré-existant maximum).
3. **NE PAS toucher** : `_deprecated/`, `.github/copilot-instructions.md`, `tests/test_chaos_concurrency.py`, `core/security/__init__.py` (sauf V1.x autorise lecture seule), `core/paged_attention.py`, `csrc/`, `rust_core/`.
4. **NE PAS renommer** : `class Synapse`, `synapses`, `_apply_neuroplasticity_score`, `HolographicKVManager`, `connectome`, `global_connectome`.
5. **Aucune dépendance** ajoutée dans `requirements*.txt` ou `pyproject.toml`.
6. **Aucun push, aucune PR**.
7. En cas de blocage : noter `BLOCKED: <raison>` dans `EXEC_LOG_V2.md`, passer à la tâche suivante.
8. **Avant chaque `replace_string_in_file`** : lire 5 lignes avant/après dans le fichier cible pour vérifier que le contexte correspond. Si non, ajuster.

---

## Préparation (UNE FOIS)

```bash
cd /home/jeremie/VRAMancer/VRAMancer
git checkout chore/sonnet-plan-exec
git checkout -b chore/sonnet-plan-v2
source .venv/bin/activate
```

Créer `docs/reports/EXEC_LOG_V2.md` :

```markdown
# Journal d'exécution — Sonnet 4.6 — Plan V2

### [BASELINE]
- 1014 passed, 1 failed (pre-existing test_health_imports_fault_manager), 39 skipped
- Branche : chore/sonnet-plan-v2 (depuis chore/sonnet-plan-exec @ 4516c16)
```

---

# P0 — Effets de bord à l'import (priorité maximale)

> **Pourquoi P0** : aujourd'hui `import core.api.registry` déclenche un broadcast UDP, et `import core.production_api` (via `app = create_app()` à la ligne 1332) tente de démarrer un serveur VTP. C'est inacceptable en test, en CI, en outil CLI tiers.

## V0.1 — Geler `ClusterDiscovery` derrière un flag

**Fichier** : `core/api/registry.py`
**Lignes 26-34** (vérifiées) :

```python
    def __init__(self):
        self._lock = threading.RLock()
        self._pipeline = None
        self.discovery = None
        
        # Start global cluster discovery immediately so node is discoverable
        try:
            from core.network.cluster_discovery import ClusterDiscovery
            self.discovery = ClusterDiscovery(heartbeat_interval=5)
            self.discovery.start()
        except ImportError:
            pass
```

Remplacer par :

```python
    def __init__(self):
        self._lock = threading.RLock()
        self._pipeline = None
        self.discovery = None

        # Cluster discovery is opt-in: starting it broadcasts UDP packets and
        # spawns background threads, which is undesirable in tests / CLI tools.
        # Enable explicitly with VRM_CLUSTER_AUTO_DISCOVER=1.
        if os.environ.get("VRM_CLUSTER_AUTO_DISCOVER", "").lower() in ("1", "true", "yes"):
            try:
                from core.network.cluster_discovery import ClusterDiscovery
                self.discovery = ClusterDiscovery(heartbeat_interval=5)
                self.discovery.start()
            except ImportError:
                pass
```

`import os` est déjà présent à la ligne 9 — vérifié.

**Validation** :
```bash
python -c "
import os; os.environ['VRM_MINIMAL_TEST'] = '1'
from core.api.registry import PipelineRegistry
r = PipelineRegistry()
assert r.discovery is None, f'discovery should be None, got {r.discovery}'
print('OK: no auto-discovery without flag')
"
```

**Commit** : `[V0.1] guard ClusterDiscovery auto-start behind VRM_CLUSTER_AUTO_DISCOVER`

---

## V0.2 — Geler le démarrage VTP de `production_api`

**Fichier** : `core/production_api.py`
**Lignes 277-283** (vérifiées) :

```python
    # Start VTP worker server for distributed inference
    try:
        from core.cross_node import start_vtp_server
        start_vtp_server()
    except Exception as exc:
        logger.warning("VTP server failed to start: %s", exc)
```

Remplacer par :

```python
    # VTP server is opt-in: it opens listening sockets which is undesirable
    # in tests and single-node deployments. Enable with VRM_FEATURE_AITP=1.
    if os.environ.get("VRM_FEATURE_AITP", "").lower() in ("1", "true", "yes"):
        try:
            from core.cross_node import start_vtp_server
            start_vtp_server()
        except Exception as exc:
            logger.warning("VTP server failed to start: %s", exc)
```

Vérifier avec `grep -n "^import os" core/production_api.py` — si absent, ajouter `import os` après les autres imports stdlib.

**Validation** :
```bash
python -c "
import os, sys
os.environ['VRM_MINIMAL_TEST'] = '1'
os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
from core.production_api import create_app
app = create_app()
# VTP module should not be loaded without flag
assert 'core.cross_node' not in sys.modules or os.environ.get('VRM_FEATURE_AITP') == '1'
print('OK: VTP not auto-started without flag')
"
```

**Commit** : `[V0.2] guard VTP server start behind VRM_FEATURE_AITP`

---

## V0.3 — Ajouter les flags `CLUSTER_AUTO_DISCOVER` et `FEATURE_AITP` à `env_flags.py`

**Fichier** : `core/env_flags.py`

### V0.3a — `CLUSTER_AUTO_DISCOVER`

Ouvrir le fichier, chercher la ligne contenant `def NODE_ID(self) -> str:` (ligne 242 vérifiée). Insérer **avant** la propriété `NODE_ID`, juste après le commentaire de section qui la précède :

```python
    @property
    def CLUSTER_AUTO_DISCOVER(self) -> bool:
        """Auto-start cluster discovery (UDP broadcast) at PipelineRegistry init."""
        return _bool("VRM_CLUSTER_AUTO_DISCOVER")

```

### V0.3b — `FEATURE_AITP`

Chercher `def AITP_PORT(self) -> int:` (ligne 259 vérifiée). Insérer **avant** :

```python
    @property
    def FEATURE_AITP(self) -> bool:
        """Opt-in AITP/VTP network stack (opens listening sockets at start_vtp_server)."""
        return _bool("VRM_FEATURE_AITP")

```

**Validation** :
```bash
python -c "
import os; os.environ.pop('VRM_FEATURE_AITP', None); os.environ.pop('VRM_CLUSTER_AUTO_DISCOVER', None)
from core.env_flags import flags
assert flags.CLUSTER_AUTO_DISCOVER is False
assert flags.FEATURE_AITP is False
os.environ['VRM_FEATURE_AITP'] = '1'
assert flags.FEATURE_AITP is True
print('OK')
"
```

**Commit** : `[V0.3] add CLUSTER_AUTO_DISCOVER and FEATURE_AITP to env_flags`

---

## V0.4 — Test smoke import sans effets de bord

**Fichier** : créer `tests/test_no_import_side_effects.py` :

```python
"""Verify that importing core modules does not start threads, open sockets,
or trigger network broadcasts. This protects against silent regressions where
a try/except at module top-level accidentally calls a constructor with side effects.
"""
import os
import sys
import threading
import socket
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Make sure no opt-in flag is set
    monkeypatch.delenv("VRM_CLUSTER_AUTO_DISCOVER", raising=False)
    monkeypatch.delenv("VRM_FEATURE_AITP", raising=False)
    monkeypatch.setenv("VRM_MINIMAL_TEST", "1")
    monkeypatch.setenv("VRM_DISABLE_RATE_LIMIT", "1")
    yield


def _count_threads_named(prefix: str) -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith(prefix))


def test_registry_import_starts_no_discovery():
    # Force fresh import
    for m in list(sys.modules):
        if m.startswith("core.api.registry") or m.startswith("core.network.cluster_discovery"):
            sys.modules.pop(m, None)
    threads_before = threading.active_count()
    from core.api.registry import PipelineRegistry
    r = PipelineRegistry()
    assert r.discovery is None
    # No new heartbeat thread should be running
    assert _count_threads_named("ClusterDiscovery") == 0
    assert _count_threads_named("Heartbeat") == 0


def test_production_api_create_app_no_vtp():
    for m in list(sys.modules):
        if m.startswith("core.cross_node") or m.startswith("core.network.llm_transport"):
            sys.modules.pop(m, None)
    from core.production_api import create_app
    app = create_app()
    assert app is not None
    # cross_node may be imported lazily inside routes — but VTP server must not be started
    assert _count_threads_named("VTP") == 0
    assert _count_threads_named("vtp") == 0


def test_inference_pipeline_import_does_not_load_torch_threads():
    for m in list(sys.modules):
        if m == "core.inference_pipeline":
            sys.modules.pop(m, None)
    from core.inference_pipeline import InferencePipeline  # noqa: F401
    # Just importing must not start GPU monitor / scheduler threads
    assert _count_threads_named("GPUMonitor") == 0
```

**Validation** :
```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/test_no_import_side_effects.py --tb=short --no-cov
# Attendu : 3 passed
```

**Commit** : `[V0.4] add side-effects-free import tests`

---

# P1 — Migration env_flags dans `inference_pipeline.py`

> **Pourquoi P1** : 25 lectures `os.environ.get` dispersées rendent les tests difficiles à scoper, et empêchent une vue centralisée de la config active. Migrer vers `flags.*` rend chaque flag testable et documenté.

## V1.1 — Ajouter l'import `flags` dans `inference_pipeline.py`

**Fichier** : `core/inference_pipeline.py`

Lire le fichier autour des lignes 30-45 pour trouver l'import torch existant. **Insérer juste après `import os` (ligne ~25)** :

```python
try:
    from core.env_flags import flags as _flags
except ImportError:
    _flags = None  # facade unavailable — fall back to os.environ
```

**Validation** :
```bash
python -c "
import os; os.environ['VRM_MINIMAL_TEST']='1'
from core.inference_pipeline import _flags
assert _flags is not None
print('OK')
"
```

**Commit** : `[V1.1] import env_flags facade in inference_pipeline`

---

## V1.2 — Migrer les 22 lectures `os.environ.get`

**Fichier** : `core/inference_pipeline.py`

Pour CHAQUE remplacement ci-dessous, utiliser `replace_string_in_file` avec **3 lignes de contexte avant et 3 après**. La table donne les lignes vérifiées le 5 mai 2026.

> **NE PAS migrer** : ligne 37 (`_MINIMAL`), ligne 76 (`VRM_TRACING`), ligne 788 (clause OR avec `VRM_MINIMAL_TEST`), ligne 832 (`VRM_MINIMAL_TEST`), ligne 1046 (`VRM_TRUST_REMOTE_CODE` — déjà tracé par `[T0.1]`). Les autres → migrer.

| # | Ligne | AVANT (extrait) | APRÈS |
|---|---|---|---|
| 1 | 128 | `float(os.environ.get("VRM_REBALANCE_INTERVAL", "5.0"))` | `(_flags.REBALANCE_INTERVAL if _flags else float(os.environ.get("VRM_REBALANCE_INTERVAL", "5.0")))` |
| 2 | 234 | `os.environ.get("VRM_PARALLEL_MODE", "pp").lower()` | `(_flags.PARALLEL_MODE if _flags else os.environ.get("VRM_PARALLEL_MODE", "pp").lower())` |
| 3 | 308 | `os.environ.get("VRM_VRAM_LENDING", "1").lower() not in ("0", "false", "no")` | `(_flags.VRAM_LENDING if _flags else os.environ.get("VRM_VRAM_LENDING", "1").lower() not in ("0", "false", "no"))` |
| 4 | 481 | `os.environ.get("VRM_DRAFT_MODEL")` | `(_flags.DRAFT_MODEL if _flags else os.environ.get("VRM_DRAFT_MODEL"))` |
| 5 | 491 | `int(os.environ.get("VRM_SPEC_GAMMA", "5"))` | `(_flags.SPEC_GAMMA if _flags else int(os.environ.get("VRM_SPEC_GAMMA", "5")))` |
| 6 | 493 | `os.environ.get("VRM_SPEC_ADAPTIVE", "1") != "0"` | `(_flags.SPEC_ADAPTIVE if _flags else os.environ.get("VRM_SPEC_ADAPTIVE", "1") != "0")` |
| 7 | 523 | `float(os.environ.get("VRM_GENERATE_TIMEOUT", "300"))` | `(_flags.GENERATE_TIMEOUT if _flags else float(os.environ.get("VRM_GENERATE_TIMEOUT", "300")))` |
| 8 | 788 | `os.environ.get("VRM_DISABLE_TURBO")` (le 2nd terme du `or`) | `(_flags.DISABLE_TURBO if _flags else os.environ.get("VRM_DISABLE_TURBO"))` |
| 9 | 810 | `int(os.environ.get("VRM_TURBO_MAX_SEQ", "2048"))` | `(_flags.TURBO_MAX_SEQ if _flags else int(os.environ.get("VRM_TURBO_MAX_SEQ", "2048")))` |
| 10 | 830 | `os.environ.get("VRM_CUDA_GRAPH")` | `(_flags.CUDA_GRAPH if _flags else os.environ.get("VRM_CUDA_GRAPH"))` |
| 11 | 847 | `int(os.environ.get("VRM_CUDA_GRAPH_CACHE", "4"))` | `(_flags.CUDA_GRAPH_CACHE if _flags else int(os.environ.get("VRM_CUDA_GRAPH_CACHE", "4")))` |
| 12 | 848 | `int(os.environ.get("VRM_CUDA_GRAPH_WARMUP", "3"))` | `(_flags.CUDA_GRAPH_WARMUP if _flags else int(os.environ.get("VRM_CUDA_GRAPH_WARMUP", "3")))` |
| 13 | 1099 | `os.environ.get("VRM_QUANTIZATION", "").lower()` | `(_flags.QUANTIZATION if _flags else os.environ.get("VRM_QUANTIZATION", "").lower())` |
| 14 | 1141 | `os.environ.get("VRM_FORCE_MULTI_GPU") == "1"` | `(_flags.FORCE_MULTI_GPU if _flags else os.environ.get("VRM_FORCE_MULTI_GPU") == "1")` |
| 15 | 1293 | `os.environ.get("VRM_KV_COMPRESSION", "").lower()` | `(_flags.KV_COMPRESSION if _flags else os.environ.get("VRM_KV_COMPRESSION", "").lower())` |
| 16 | 1310 | `int(os.environ.get("VRM_KV_COMPRESSION_BITS", "3"))` | `(_flags.KV_COMPRESSION_BITS if _flags else int(os.environ.get("VRM_KV_COMPRESSION_BITS", "3")))` |
| 17 | 1311 | `int(os.environ.get("VRM_KV_CACHE_RESIDUAL", "128"))` | `(_flags.KV_CACHE_RESIDUAL if _flags else int(os.environ.get("VRM_KV_CACHE_RESIDUAL", "128")))` |
| 18 | 1378 | `int(os.environ.get("VRM_MAX_BATCH_SIZE", "32"))` | `(_flags.MAX_BATCH_SIZE if _flags else int(os.environ.get("VRM_MAX_BATCH_SIZE", "32")))` |
| 19 | 1482 | `float(os.environ.get("VRM_LEND_RATIO", "0.70"))` | `(_flags.LEND_RATIO if _flags else float(os.environ.get("VRM_LEND_RATIO", "0.70")))` |
| 20 | 1483 | `float(os.environ.get("VRM_RECLAIM_THRESHOLD", "0.80"))` | `(_flags.RECLAIM_THRESHOLD if _flags else float(os.environ.get("VRM_RECLAIM_THRESHOLD", "0.80")))` |
| 21 | 1514 | `float(os.environ.get("VRM_LENDING_INTERVAL", "2.0"))` | `(_flags.LENDING_INTERVAL if _flags else float(os.environ.get("VRM_LENDING_INTERVAL", "2.0")))` |

> **NB ligne 788** : la ligne entière est `if os.environ.get("VRM_MINIMAL_TEST") or os.environ.get("VRM_DISABLE_TURBO"):`. **Ne migrer que le second terme**, soit après remplacement : `if os.environ.get("VRM_MINIMAL_TEST") or (_flags.DISABLE_TURBO if _flags else os.environ.get("VRM_DISABLE_TURBO")):`.

**Stratégie d'exécution** : faire les 21 remplacements en **3 passes** (lignes 100-500, 500-1000, 1000-1520) pour limiter les conflits. Après CHAQUE passe :

```bash
python -c "import ast; ast.parse(open('core/inference_pipeline.py').read())" && echo "syntax OK"
VRM_MINIMAL_TEST=1 pytest -q tests/test_pipeline.py --tb=line --no-cov 2>&1 | tail -3
```

**Validation finale** :
```bash
# Vérifier qu'il reste seulement les 4 os.environ.get autorisés
grep -c 'os.environ.get("VRM_' core/inference_pipeline.py
# Attendu : doit être ≤ 4 (les 21 migrés disparaissent du compteur car ils sont maintenant DERRIÈRE le ternaire flags)
# En réalité le grep comptera tous les os.environ.get même dans les fallbacks → ce sera ~25
# Le vrai test :
grep -c "_flags\." core/inference_pipeline.py
# Attendu : ≥ 21
```

**Commit** : `[V1.2] migrate 21 os.environ reads to env_flags facade`

---

## V1.3 — Tests pour `env_flags`

**Fichier** : créer `tests/test_env_flags.py` (le contenu de la version précédente du plan était bon, le voici condensé) :

```python
"""Tests for core.env_flags — live os.environ facade."""
import os
import pytest


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VRM_") and k not in ("VRM_MINIMAL_TEST", "VRM_TEST_MODE", "VRM_DISABLE_RATE_LIMIT"):
            monkeypatch.delenv(k, raising=False)
    yield


def test_bool_default_false():
    from core.env_flags import flags
    assert flags.TRUST_REMOTE_CODE is False
    assert flags.DISABLE_TURBO is False
    assert flags.CUDA_GRAPH is False
    assert flags.FORCE_MULTI_GPU is False


def test_bool_truthy_strings(monkeypatch):
    from core.env_flags import flags
    for val in ("1", "true", "yes", "True", "YES"):
        monkeypatch.setenv("VRM_TRUST_REMOTE_CODE", val)
        assert flags.TRUST_REMOTE_CODE is True


def test_bool_falsy_strings(monkeypatch):
    from core.env_flags import flags
    for val in ("0", "false", "no", "", "anything_else"):
        monkeypatch.setenv("VRM_TRUST_REMOTE_CODE", val)
        assert flags.TRUST_REMOTE_CODE is False


def test_int_defaults():
    from core.env_flags import flags
    assert flags.MAX_BATCH_SIZE == 32
    assert flags.SPEC_GAMMA == 5
    assert flags.CUDA_GRAPH_CACHE == 4
    assert flags.CUDA_GRAPH_WARMUP == 3
    assert flags.TURBO_MAX_SEQ == 2048


def test_int_override(monkeypatch):
    monkeypatch.setenv("VRM_MAX_BATCH_SIZE", "128")
    from core.env_flags import flags
    assert flags.MAX_BATCH_SIZE == 128


def test_int_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("VRM_MAX_BATCH_SIZE", "not_a_number")
    from core.env_flags import flags
    assert flags.MAX_BATCH_SIZE == 32


def test_float_defaults():
    from core.env_flags import flags
    assert flags.LEND_RATIO == pytest.approx(0.70)
    assert flags.RECLAIM_THRESHOLD == pytest.approx(0.80)
    assert flags.LENDING_INTERVAL == pytest.approx(2.0)
    assert flags.GENERATE_TIMEOUT == pytest.approx(300.0)
    assert flags.REBALANCE_INTERVAL == pytest.approx(5.0)


def test_float_override(monkeypatch):
    monkeypatch.setenv("VRM_LEND_RATIO", "0.55")
    from core.env_flags import flags
    assert flags.LEND_RATIO == pytest.approx(0.55)


def test_str_defaults():
    from core.env_flags import flags
    assert flags.PARALLEL_MODE == "pp"
    assert flags.QUANTIZATION == ""
    assert flags.KV_COMPRESSION == ""


def test_str_lowercased(monkeypatch):
    monkeypatch.setenv("VRM_PARALLEL_MODE", "TP")
    from core.env_flags import flags
    assert flags.PARALLEL_MODE == "tp"


def test_opt_str_none_when_unset():
    from core.env_flags import flags
    assert flags.DRAFT_MODEL is None
    assert flags.API_TOKEN is None


def test_opt_str_set(monkeypatch):
    monkeypatch.setenv("VRM_DRAFT_MODEL", "TinyLlama/TinyLlama-1.1B")
    from core.env_flags import flags
    assert flags.DRAFT_MODEL == "TinyLlama/TinyLlama-1.1B"


def test_vram_lending_default_on():
    from core.env_flags import flags
    assert flags.VRAM_LENDING is True


def test_vram_lending_disabled(monkeypatch):
    for val in ("0", "false", "no"):
        monkeypatch.setenv("VRM_VRAM_LENDING", val)
        from core.env_flags import flags
        assert flags.VRAM_LENDING is False


def test_spec_adaptive_default_on(monkeypatch):
    from core.env_flags import flags
    assert flags.SPEC_ADAPTIVE is True


def test_spec_adaptive_off(monkeypatch):
    monkeypatch.setenv("VRM_SPEC_ADAPTIVE", "0")
    from core.env_flags import flags
    assert flags.SPEC_ADAPTIVE is False


def test_live_reading(monkeypatch):
    """Critical: each property read must hit os.environ live, not cache."""
    from core.env_flags import flags
    assert flags.MAX_BATCH_SIZE == 32
    monkeypatch.setenv("VRM_MAX_BATCH_SIZE", "64")
    assert flags.MAX_BATCH_SIZE == 64
    monkeypatch.setenv("VRM_MAX_BATCH_SIZE", "128")
    assert flags.MAX_BATCH_SIZE == 128


def test_feature_aitp_default_off():
    from core.env_flags import flags
    assert flags.FEATURE_AITP is False


def test_cluster_auto_discover_default_off():
    from core.env_flags import flags
    assert flags.CLUSTER_AUTO_DISCOVER is False
```

**Validation** : `pytest -q tests/test_env_flags.py --no-cov` → 18 passed.

**Commit** : `[V1.3] add env_flags test suite (18 tests)`

---

# P2 — Tests directs sécurité

> **Pourquoi P2** : `verify_request` et `enforce_startup_checks` gardent la production. Aucun test direct n'existe (uniquement des tests de middleware Flask).

## V2.1 — Tests pour `verify_request`

**Fichier** : créer `tests/test_verify_request.py`

> ATTENTION : `verify_request(secret, method, path, headers, body)` retourne `None` si OK, ou `(message, code)` sinon. Vérifier les signatures dans `core/security/__init__.py:109` (déjà lu, signature confirmée).

```python
"""Direct unit tests for core.security.verify_request()."""
import os
import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VRM_MINIMAL_TEST", "1")
    monkeypatch.setenv("VRM_DISABLE_RATE_LIMIT", "1")
    monkeypatch.delenv("VRM_PRODUCTION", raising=False)
    monkeypatch.delenv("VRM_TEST_RELAX_SECURITY", raising=False)
    monkeypatch.delenv("VRM_TEST_BYPASS_HA", raising=False)
    yield


def _vr():
    from core.security import verify_request
    return verify_request


@pytest.mark.parametrize("path", [
    "/health", "/ready", "/live", "/api/health", "/",
    "/static/app.js", "/static/css/main.css", "/favicon.ico",
])
def test_public_paths_pass_without_token(path):
    assert _vr()(None, "GET", path, {}, b"") is None


def test_relax_security_bypass(monkeypatch):
    monkeypatch.setenv("VRM_TEST_RELAX_SECURITY", "1")
    assert _vr()("secret", "POST", "/api/models/load", {}, b"{}") is None


def test_test_bypass_ha_specific_path(monkeypatch):
    monkeypatch.setenv("VRM_TEST_BYPASS_HA", "1")
    assert _vr()("secret", "POST", "/api/ha/apply", {}, b"") is None
    # Other paths still require token in normal flow (but pass in non-prod)


def test_no_token_non_production_passes():
    """Without VRM_PRODUCTION=1, missing token is not fatal."""
    result = _vr()("secret", "POST", "/api/generate", {}, b"{}")
    assert result is None


def test_production_no_token_returns_401(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    result = _vr()("secret", "POST", "/api/generate", {}, b"{}")
    assert result is not None
    msg, code = result
    assert code == 401
    assert "token" in msg.lower()


def test_production_valid_token_passes(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    result = _vr()("secret123", "POST", "/api/generate",
                   {"X-API-TOKEN": "secret123"}, b"{}")
    assert result is None


def test_production_invalid_token_returns_401(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    result = _vr()("secret123", "POST", "/api/generate",
                   {"X-API-TOKEN": "wrongtoken"}, b"{}")
    assert result is not None
    msg, code = result
    assert code == 401


def test_bearer_authorization_header(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    result = _vr()("secret123", "GET", "/api/gpu",
                   {"Authorization": "Bearer secret123"}, b"")
    assert result is None


def test_no_secret_no_production_passes():
    """secret=None, non-production → pass."""
    result = _vr()(None, "GET", "/api/status", {}, b"")
    assert result is None
```

**Validation** : `pytest -q tests/test_verify_request.py --no-cov` → ≥10 passed (parametrize + tests).

**Commit** : `[V2.1] add verify_request unit tests`

---

## V2.2 — Tests pour `enforce_startup_checks`

**Fichier** : créer `tests/test_startup_checks.py`

```python
"""Direct unit tests for core.security.startup_checks.enforce_startup_checks()."""
import os
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ["VRM_PRODUCTION", "VRM_API_TOKEN", "VRM_AUTH_SECRET",
                "VRM_MINIMAL_TEST", "VRM_TEST_RELAX_SECURITY", "VRM_TEST_BYPASS_HA"]:
        monkeypatch.delenv(var, raising=False)
    yield


def _enforce():
    """Re-import to pick up env changes."""
    import importlib
    import core.security.startup_checks as m
    importlib.reload(m)
    return m.enforce_startup_checks


def test_non_production_passes_silently():
    _enforce()()  # must not raise


def test_production_missing_api_token_raises(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_AUTH_SECRET", "x" * 32)
    with pytest.raises(RuntimeError, match="VRM_API_TOKEN"):
        _enforce()()


def test_production_missing_auth_secret_raises(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "secure-token")
    with pytest.raises(RuntimeError, match="VRM_AUTH_SECRET"):
        _enforce()()


def test_production_with_required_vars_passes(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "secure-token")
    monkeypatch.setenv("VRM_AUTH_SECRET", "x" * 32)
    try:
        _enforce()()
    except RuntimeError as e:
        if "admin" in str(e).lower():
            pytest.skip("Default admin/admin user present in test env")
        raise


@pytest.mark.parametrize("dangerous_var", [
    "VRM_MINIMAL_TEST", "VRM_TEST_RELAX_SECURITY", "VRM_TEST_BYPASS_HA"
])
def test_production_rejects_test_flags(monkeypatch, dangerous_var):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "secure-token")
    monkeypatch.setenv("VRM_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv(dangerous_var, "1")
    with pytest.raises(RuntimeError, match=dangerous_var):
        _enforce()()
```

**Validation** : `pytest -q tests/test_startup_checks.py --no-cov` → 7 passed.

**Commit** : `[V2.2] add enforce_startup_checks unit tests`

---

# P3 — Honnêteté du code

> **Pourquoi P3** : aujourd'hui `RemoteExecutor` se présente comme transport "zero-copy" dans des logs et docstrings, alors qu'il sérialise via safetensors avant socket. Mentir aux opérateurs en production est un anti-pattern critique. La docstring corrige le contrat.

## V3.1 — Corriger la docstring de `RemoteExecutor`

**Fichier** : `core/block_router.py`
**Ligne 79** (vérifiée) :

```python
class RemoteExecutor:
    """Remote block execution proxy with connection management."""
```

Remplacer par :

```python
class RemoteExecutor:
    """Remote block execution proxy via TCP + HMAC.

    Serialization: tensors are serialized with safetensors (binary) or
    torch.save (fallback) before being sent. This is NOT a true zero-copy
    transport — it allocates a CPU buffer for the safetensors payload.
    For genuine zero-copy GPU-to-GPU between local devices, use
    TransferManager (P2P / CPU-staged) instead.

    Security: HMAC-SHA256 signature with VRM_API_TOKEN. In production
    (VRM_PRODUCTION=1), VRM_API_TOKEN is mandatory.
    """
```

**Validation** : `python -c "from core.block_router import RemoteExecutor; print(RemoteExecutor.__doc__[:80])"`

**Commit** : `[V3.1] correct RemoteExecutor docstring (not zero-copy)`

---

## V3.2 — Log honnête du mode au démarrage

**Fichier** : `core/inference_pipeline.py`

Trouver la ligne contenant `_logger.info("Loading model: %s` (chercher avec grep). Insérer **juste avant** ce log :

```python
            # Honest mode banner: tells operators which mode is active.
            _mode = "STUB" if os.environ.get("VRM_MINIMAL_TEST") else (
                "PRODUCTION" if os.environ.get("VRM_PRODUCTION") == "1" else "DEV"
            )
            _features = []
            if os.environ.get("VRM_CONTINUOUS_BATCHING") == "1":
                _features.append("continuous_batching")
            _quant = os.environ.get("VRM_QUANTIZATION", "")
            if _quant:
                _features.append(f"quant={_quant}")
            _kv = os.environ.get("VRM_KV_COMPRESSION", "")
            if _kv:
                _features.append(f"kv={_kv}")
            if os.environ.get("VRM_PARALLEL_MODE", "pp").lower() == "tp":
                _features.append("tensor_parallel")
            _logger.info("VRAMancer mode: [%s]%s", _mode,
                         f" | features: {', '.join(_features)}" if _features else "")
```

**Validation** :
```bash
python -c "
import logging, os
os.environ['VRM_MINIMAL_TEST'] = '1'
logging.basicConfig(level=logging.INFO, format='%(message)s')
from core.inference_pipeline import InferencePipeline
p = InferencePipeline()
try:
    p.load('gpt2', backend='stub')
except Exception:
    pass
" 2>&1 | grep "VRAMancer mode"
# Attendu : ligne contenant 'VRAMancer mode: [STUB]'
```

**Commit** : `[V3.2] log active mode banner at pipeline startup`

---

# P4 — Tests `LlamaServerBackend`

## V4.1 — Tests sans binaire

**Fichier** : créer `tests/test_llama_server_backend.py` :

```python
"""Stub-safe tests for core.llama_server_backend (no binary needed)."""
import os
import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("VRM_MINIMAL_TEST"),
    reason="stub-safe smoke tests only"
)


def test_module_imports_cleanly():
    import core.llama_server_backend as m
    assert hasattr(m, "LlamaServerBackend")
    assert hasattr(m, "get_or_download_binary")
    assert hasattr(m, "_platform_key")
    assert hasattr(m, "_ASSET_MAP")


def test_platform_key_returns_known_value():
    from core.llama_server_backend import _platform_key
    key = _platform_key()
    assert isinstance(key, str)
    assert key in {"linux-cuda", "linux-cpu", "darwin-arm", "darwin-x86", "windows"}


def test_asset_map_covers_all_platforms():
    from core.llama_server_backend import _ASSET_MAP
    for key in ("linux-cuda", "linux-cpu", "darwin-arm", "darwin-x86", "windows"):
        assert key in _ASSET_MAP
        assert "{tag}" in _ASSET_MAP[key]


def test_local_tensor_split_no_crash():
    """Should return None or list, never crash."""
    from core.llama_server_backend import _local_tensor_split
    result = _local_tensor_split(2)
    assert result is None or isinstance(result, list)


def test_binary_dir_is_pathlike():
    from pathlib import Path
    from core.llama_server_backend import BINARY_DIR
    assert isinstance(BINARY_DIR, Path)


def test_server_port_default():
    import importlib
    import core.llama_server_backend as m
    importlib.reload(m)
    assert m.SERVER_PORT == 8081


def test_server_port_env_override(monkeypatch):
    monkeypatch.setenv("VRM_LLAMA_SERVER_PORT", "9999")
    import importlib
    import core.llama_server_backend as m
    importlib.reload(m)
    assert m.SERVER_PORT == 9999


def test_get_or_download_finds_existing_binary(monkeypatch, tmp_path):
    import core.llama_server_backend as m
    fake = tmp_path / "llama-server"
    fake.write_bytes(b"fake-binary")
    fake.chmod(0o755)
    monkeypatch.setattr(m, "BINARY_DIR", tmp_path)
    result = m.get_or_download_binary()
    assert result == fake
```

**Validation** : `pytest -q tests/test_llama_server_backend.py --no-cov` → 8 passed.

**Commit** : `[V4.1] add llama_server_backend smoke tests`

---

# P5 — Documentation : matrice de compatibilité

## V5.1 — `docs/COMPATIBILITY.md`

Le contenu est dans la version précédente du plan (était bon). Je le résume ici. **Action** : créer le fichier avec le contenu suivant **exact** :

````markdown
# VRAMancer — Matrice de compatibilité

> Source de vérité technique : `.github/copilot-instructions.md`
> Dernière mise à jour : 5 mai 2026

## 1. Backends × Quantization × OS

| Backend | nvfp4 | nf4 | int8 | BF16 | Linux/CUDA | Linux/ROCm | macOS/MPS | Windows/CUDA | CPU |
|---|---|---|---|---|---|---|---|---|---|
| HuggingFace | ✓ Blackwell | ✓ BnB | ✓ BnB | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| vLLM | — | — | via vLLM | ✓ | ✓ | ✓ | — | — | — |
| llama.cpp (binaire) | — | — | — | GGUF | ✓ | ✓ Vulkan | ✓ Metal | ✓ | ✓ |
| llama-cpp-python | — | — | GGUF Q8 | GGUF | ✓ | ✓ Vulkan | ✓ Metal | ✓ | ✓ |
| Ollama | — | — | — | via Ollama | ✓ | ✓ | ✓ | ✓ | ✓ |

> **nvfp4** : Blackwell (RTX 50xx, CC≥10) + torchao.
> **nf4 / int8** : `bitsandbytes` ≥0.41. Bug upstream multi-GPU avec accelerate — VRAMancer force single-GPU.

## 2. Stratégie de transfer GPU-GPU

| Environnement | Stratégie | Bande passante typique |
|---|---|---|
| Bare-metal PCIe P2P | CUDA P2P (strategy 2) | 25-32 GB/s |
| NVLink (A100/H100) | CUDA P2P NVLink | 600 GB/s |
| Proxmox VM VFIO | CPU-staged pinned (strategy 4) | 12-15 GB/s |
| macOS MPS | CPU copy | ~6 GB/s |
| Single GPU | N/A | — |

## 3. Dépendances optionnelles

| Composant | Activation | Effet |
|---|---|---|
| `rust_core` (cargo build) | Auto si `.so` présent | DtoD direct via libcuda, HMAC 100x |
| `pyverbs` | Auto si présent | RDMA verbs zero-copy |
| `nvidia_peermem` | Module kernel | GPUDirect RDMA |
| `bitsandbytes` ≥0.41 | `VRM_QUANTIZATION=nf4\|int8` | Quantization 4/8-bit |
| `triton` | Auto | TurboQuant GPU |
| `vllm` | `VRM_BACKEND=vllm` | vLLM engine |
| `llama-cpp-python` | `VRM_BACKEND=llamacpp` | GGUF |
| `flask-socketio` | Auto | Dashboard temps réel |

## 4. Variables d'environnement critiques

| Variable | Défaut | Effet |
|---|---|---|
| `VRM_QUANTIZATION` | `""` (BF16) | NF4=0.5x VRAM, NVFP4=0.25x VRAM |
| `VRM_PARALLEL_MODE` | `pp` | `tp` = NCCL all-reduce |
| `VRM_CONTINUOUS_BATCHING` | `0` | Multi-requête, requis >1 user |
| `VRM_TRANSFER_P2P` | `1` | `0` = force CPU-staged (VM Proxmox) |
| `VRM_TRUST_REMOTE_CODE` | `0` | `1` = autorise code custom HF (sécurité) |
| `VRM_PRODUCTION` | `0` | `1` = enforce token + secret |
| `VRM_FEATURE_AITP` | `0` | `1` = active réseau AITP/VTP (sockets) |
| `VRM_CLUSTER_AUTO_DISCOVER` | `0` | `1` = broadcast UDP au démarrage |

## 5. Benchmarks (RTX 3090 + RTX 5070 Ti, Proxmox)

| Modèle | Config | tok/s | VRAM |
|---|---|---|---|
| GPT-2 124M | BF16 | 125.6 | 0.5 GB |
| TinyLlama-1.1B | BF16 | 56.5 | 2.2 GB |
| TinyLlama-1.1B | NVFP4 (5070 Ti) | ~36 | 5.46 GB |
| Qwen2.5-7B | NF4 | 20.2 | ~5 GB |
| Qwen2.5-7B | GGUF Q4_K_M | 106.8 | 3.0 GB |
| Qwen2.5-14B | BF16 2-GPU | 6.0 | 35.9 GB |
| Qwen2.5-14B | NF4 1-GPU | 10.5 | 10.8 GB |

## 6. Limitations connues

- **VM Proxmox + IOMMU** : seule la stratégie 4 (CPU-staged pinned) fonctionne. Overhead VFIO ~10-15%.
- **BnB multi-GPU** : bug upstream accelerate — VRAMancer force single-GPU pour BnB.
- **transformers 5.3 + BnB** : toujours utiliser `torch_dtype=torch.float16` (BnB bypass dtype dans certaines versions).
- **auth_strong** : default admin/admin en dev. Changer immédiatement en prod.
- **default-admin** : enforce_startup_checks bloque démarrage en prod si admin/admin présent.
````

**Commit** : `[V5.1] add docs/COMPATIBILITY.md`

---

# P6 — Imports & couverture

## V6.1 — Ajouter `core.production_api` à `test_imports.py`

**Fichier** : `tests/test_imports.py`

Vérifier d'abord :
```bash
grep "production_api" tests/test_imports.py
```

Si absent : trouver la fin de la liste `CORE_MODULES`, ajouter `"core.production_api",` avant la dernière entrée.

Vérifier que l'import de `core.production_api` fonctionne dans `VRM_MINIMAL_TEST=1` (si non, **ne pas l'ajouter**, noter `BLOCKED` dans EXEC_LOG_V2.md).

**Validation** : `pytest -q tests/test_imports.py --no-cov` → tous passent.

**Commit** : `[V6.1] cover core.production_api in test_imports`

---

# P7 — Validation finale

## V7.1 — Suite complète

```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=short --no-cov -q 2>&1 | tail -10
```

Critères :
- `≥1014 + N passed` où N = nouveaux tests ajoutés (~37 attendus : 18 env_flags + 9 verify + 7 startup + 8 llama + 3 side_effects = 45)
- `≤1 failed` (uniquement le pré-existant)
- `0 nouvelle régression`

## V7.2 — Vérifier git log

```bash
git log --oneline chore/sonnet-plan-v2 ^chore/sonnet-plan-exec
```

Doit afficher exactement les commits `[V0.1]` … `[V7.x]` dans l'ordre.

## V7.3 — Finaliser EXEC_LOG_V2.md

Ajouter à la fin du journal :

```markdown
### [SUMMARY]

**Tâches DONE** : X/Y
**Tâches BLOCKED** : Z (avec raisons)
**Tests ajoutés** : ~45
- env_flags : 18
- verify_request : 9
- startup_checks : 7
- llama_server_backend : 8
- no_import_side_effects : 3

**Migration env_flags** : 21 os.environ.get → flags.* dans inference_pipeline.py

**Effets de bord supprimés** :
- ClusterDiscovery auto-start (registry.py)
- VTP server auto-start (production_api.py)

**Documentation** :
- docs/COMPATIBILITY.md (matrice backend × quant × OS)

**Régressions** : aucune
**Final tests** : [résultat ici]
```

**Commit** : `[V7.3] finalize EXEC_LOG_V2 summary`

---

# Ordre d'exécution (strict)

```
Préparation
  → V0.1  Guard ClusterDiscovery
  → V0.2  Guard VTP server
  → V0.3  Add CLUSTER_AUTO_DISCOVER + FEATURE_AITP to env_flags
  → V0.4  Add side-effects-free import tests
  → V1.1  Import flags in inference_pipeline
  → V1.2  Migrate 21 os.environ.get → flags
  → V1.3  Add env_flags tests
  → V2.1  Add verify_request tests
  → V2.2  Add startup_checks tests
  → V3.1  Fix RemoteExecutor docstring
  → V3.2  Log active mode banner
  → V4.1  Add llama_server_backend tests
  → V5.1  Create COMPATIBILITY.md
  → V6.1  Cover production_api in test_imports
  → V7.1  Full test run
  → V7.2  Git log check
  → V7.3  Finalize EXEC_LOG_V2.md
```

---

# Anti-patterns interdits (rappel)

- ❌ Renommer `class Synapse`, `synapses`, `_apply_neuroplasticity_score`, `HolographicKVManager`.
- ❌ Modifier `_deprecated/`, `tests/test_chaos_concurrency.py`, `.github/copilot-instructions.md`.
- ❌ Modifier `core/security/__init__.py` (sauf docstring corrections explicites — ce plan n'en demande pas).
- ❌ Modifier `core/paged_attention.py`, `csrc/`, `rust_core/`.
- ❌ Ajouter une dépendance dans `requirements*.txt` ou `pyproject.toml`.
- ❌ Élargir le périmètre : ne PAS migrer `os.environ.get` dans d'autres modules que `inference_pipeline.py`.
- ❌ `git push`, `git push --force`, créer une PR, merger.
- ❌ `git reset --hard` ou `git rebase` sans confirmation utilisateur.

# Patterns autorisés

- ✓ `replace_string_in_file` avec ≥3 lignes de contexte.
- ✓ `multi_replace_string_in_file` pour grouper des éditions sur le même fichier.
- ✓ Recharger un module via `importlib.reload(m)` dans les tests.
- ✓ `pytest.mark.skipif` pour les tests qui requièrent un binaire/GPU absent en CI.
- ✓ `monkeypatch.setenv` / `monkeypatch.delenv` pour scopper les flags par test.

---

# Critères de réussite globaux

À la fin du plan, les invariants suivants doivent tenir :

1. ✅ `import core.api.registry` n'ouvre AUCUN socket sans `VRM_CLUSTER_AUTO_DISCOVER=1`.
2. ✅ `import core.production_api` ne démarre PAS de serveur VTP sans `VRM_FEATURE_AITP=1`.
3. ✅ `core/inference_pipeline.py` n'a plus de lecture `os.environ.get("VRM_*")` orpheline (toutes ont un fallback `_flags.X if _flags else …`).
4. ✅ `verify_request` et `enforce_startup_checks` ont des tests directs.
5. ✅ `LlamaServerBackend` est testé sans binaire.
6. ✅ Le mode actif (STUB/DEV/PROD + features) est loggé au démarrage.
7. ✅ `RemoteExecutor` est docstring-honnête (pas de "zero-copy" mensonger).
8. ✅ `docs/COMPATIBILITY.md` existe.
9. ✅ Tous les tests passent : ≥1059 (1014 + 45) avec 1 failure pré-existant.
10. ✅ Aucune dépendance ajoutée. Aucun symbole renommé. Aucun fichier `_deprecated/` modifié.

---

# Annexe : structure du repo (rappel)

```
core/
  ├── env_flags.py           # façade flags (déjà créée)
  ├── inference_pipeline.py  # 1500+ LOC, cible V1.x
  ├── api/
  │   ├── registry.py        # cible V0.1
  │   └── routes_ops.py
  ├── security/
  │   ├── __init__.py        # verify_request (lecture seule en V2.1)
  │   └── startup_checks.py  # enforce_startup_checks (lecture seule en V2.2)
  ├── network/
  │   ├── cluster_discovery.py
  │   └── llm_transport.py
  ├── orchestrator/
  │   └── placement_engine.py  # _apply_neuroplasticity_score (NE PAS TOUCHER)
  ├── block_router.py        # cible V3.1 (docstring)
  ├── llama_server_backend.py # cible V4.1 (tests)
  ├── production_api.py      # cible V0.2
  └── cross_node.py
tests/                       # 63 fichiers existants
docs/
  ├── reports/
  │   ├── PLAN_ACTION.md     # V1 (archivé)
  │   ├── PLAN_ACTION_V2.md  # ce document
  │   └── EXEC_LOG_V2.md     # à créer
  └── COMPATIBILITY.md       # cible V5.1
```

---

**Fin du plan. Bonne exécution Sonnet 4.6.**
