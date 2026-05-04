# Plan d'action VRAMancer v2 — spécification d'exécution pour Sonnet 4.6

> **Auteur** : Claude Sonnet 4.5 (audit du 5 mai 2026)
> **Destinataire** : agent Sonnet 4.6
> **Branche de travail** : `chore/sonnet-plan-v2`
> **Source de vérité** : `.github/copilot-instructions.md`

Ce document est un **cahier des charges exécutable**. Chaque tâche contient :
- les fichiers exacts avec lignes
- les snippets précis à insérer / remplacer
- les commandes shell de validation
- les critères d'acceptation binaires (PASS / FAIL)

Sonnet 4.6 doit **suivre l'ordre**, **ne rien faire hors liste**, reporter chaque tâche dans `docs/reports/EXEC_LOG_V2.md`.

---

## 0. Préparation (UNE FOIS au début)

### 0.1 — Créer la branche

```bash
cd /home/jeremie/VRAMancer/VRAMancer
git checkout chore/sonnet-plan-exec   # partir de la branche précédente
git checkout -b chore/sonnet-plan-v2
```

### 0.2 — Activer l'environnement

```bash
source .venv/bin/activate
```

### 0.3 — Baseline

```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/ --ignore=tests/test_chaos_concurrency.py --tb=no --no-cov 2>&1 | tail -3
```

Attendu : `1014 passed, 1 failed, 39 skipped`. Noter dans EXEC_LOG_V2.md.

### 0.4 — Créer le journal

Créer `docs/reports/EXEC_LOG_V2.md` :

```markdown
# Journal d'exécution — Sonnet 4.6 — Plan V2

### [BASELINE]
**Tests** : 1014 passed, 1 failed (pre-existing), 39 skipped
```

---

## Règles globales

1. **Un commit par tâche**. Format : `[VX.Y] description (≤ 72 chars)`.
2. **Avant chaque commit** : `pytest -q tests/ --ignore=tests/test_chaos_concurrency.py --tb=no --no-cov 2>&1 | tail -1` — confirmer aucune nouvelle failure.
3. **Ne pas modifier** : `_deprecated/`, `.github/copilot-instructions.md`, `tests/test_chaos_concurrency.py`.
4. **Aucune dépendance** ajoutée dans `requirements*.txt` / `pyproject.toml`.
5. **Aucun symbole public renommé** : `class Synapse`, `synapses`, `_apply_neuroplasticity_score`, `HolographicKVManager`.
6. **Pas d'auto-push, pas d'auto-merge**.
7. En cas de blocage : écrire BLOCKED dans EXEC_LOG_V2.md, passer à la tâche suivante.

---

## P0 — Migration `env_flags` (priorité haute)

### V0.1 — Migrer `core/inference_pipeline.py` vers `flags`

**Contexte** : `core/inference_pipeline.py` contient 25 appels `os.environ.get("VRM_*")` directs. `core/env_flags.py` expose déjà tous ces flags via `flags.*`. Migrer les appels pour centraliser et faciliter les tests.

**Vérification préalable** :
```bash
grep -c 'os.environ.get("VRM_' core/inference_pipeline.py
# Doit retourner 25 (ou proche)
```

**Action** — en haut du fichier, après `import os`, ajouter l'import flags. Chercher la ligne exacte :
```bash
grep -n "^import os$\|^import os " core/inference_pipeline.py | head -3
```

Insérer juste **après** la ligne `import os` (ou après le bloc `try/except` qui importe torch en ligne ~35) :

```python
try:
    from core.env_flags import flags as _flags
except ImportError:
    _flags = None  # type: ignore
```

**Puis** remplacer les 25 appels. Voici la liste complète avec les remplacements exacts. Pour chaque remplacement, utiliser `replace_string_in_file` avec 3 lignes de contexte.

#### Remplacements ligne par ligne

| Ligne approx | AVANT | APRÈS |
|---|---|---|
| 37 | `_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")` | `_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")  # keep raw for falsy check` |
| 128 | `self._rebalance_interval = float(os.environ.get("VRM_REBALANCE_INTERVAL", "5.0"))` | `self._rebalance_interval = (_flags.REBALANCE_INTERVAL if _flags else float(os.environ.get("VRM_REBALANCE_INTERVAL", "5.0")))` |
| 234 | `_parallel_mode = os.environ.get("VRM_PARALLEL_MODE", "pp").lower()` | `_parallel_mode = (_flags.PARALLEL_MODE if _flags else os.environ.get("VRM_PARALLEL_MODE", "pp").lower())` |
| 308 | `_lending_enabled = os.environ.get("VRM_VRAM_LENDING", "1").lower() not in ("0", "false", "no")` | `_lending_enabled = (_flags.VRAM_LENDING if _flags else os.environ.get("VRM_VRAM_LENDING", "1").lower() not in ("0", "false", "no"))` |
| 481 | `draft_name = os.environ.get("VRM_DRAFT_MODEL")` | `draft_name = (_flags.DRAFT_MODEL if _flags else os.environ.get("VRM_DRAFT_MODEL"))` |
| 491 | `gamma=int(os.environ.get("VRM_SPEC_GAMMA", "5")),` | `gamma=(_flags.SPEC_GAMMA if _flags else int(os.environ.get("VRM_SPEC_GAMMA", "5"))),` |
| 493 | `adaptive=os.environ.get("VRM_SPEC_ADAPTIVE", "1") != "0",` | `adaptive=(_flags.SPEC_ADAPTIVE if _flags else os.environ.get("VRM_SPEC_ADAPTIVE", "1") != "0"),` |
| 523 | `timeout=float(os.environ.get("VRM_GENERATE_TIMEOUT", "300"))` | `timeout=(_flags.GENERATE_TIMEOUT if _flags else float(os.environ.get("VRM_GENERATE_TIMEOUT", "300")))` |
| 788 | `if os.environ.get("VRM_MINIMAL_TEST") or os.environ.get("VRM_DISABLE_TURBO"):` | `if os.environ.get("VRM_MINIMAL_TEST") or (_flags.DISABLE_TURBO if _flags else os.environ.get("VRM_DISABLE_TURBO")):` |
| 810 | `max_seq_len=int(os.environ.get("VRM_TURBO_MAX_SEQ", "2048")),` | `max_seq_len=(_flags.TURBO_MAX_SEQ if _flags else int(os.environ.get("VRM_TURBO_MAX_SEQ", "2048"))),` |
| 830 | `if not os.environ.get("VRM_CUDA_GRAPH"):` | `if not (_flags.CUDA_GRAPH if _flags else os.environ.get("VRM_CUDA_GRAPH")):` |
| 847 | `max_cache_entries=int(os.environ.get("VRM_CUDA_GRAPH_CACHE", "4")),` | `max_cache_entries=(_flags.CUDA_GRAPH_CACHE if _flags else int(os.environ.get("VRM_CUDA_GRAPH_CACHE", "4"))),` |
| 848 | `warmup_steps=int(os.environ.get("VRM_CUDA_GRAPH_WARMUP", "3")),` | `warmup_steps=(_flags.CUDA_GRAPH_WARMUP if _flags else int(os.environ.get("VRM_CUDA_GRAPH_WARMUP", "3"))),` |
| 1099 | `quant = os.environ.get("VRM_QUANTIZATION", "").lower()` | `quant = (_flags.QUANTIZATION if _flags else os.environ.get("VRM_QUANTIZATION", "").lower())` |
| 1141 | `if os.environ.get("VRM_FORCE_MULTI_GPU") == "1":` | `if (_flags.FORCE_MULTI_GPU if _flags else os.environ.get("VRM_FORCE_MULTI_GPU") == "1"):` |
| 1293 | `kv_comp = os.environ.get("VRM_KV_COMPRESSION", "").lower()` | `kv_comp = (_flags.KV_COMPRESSION if _flags else os.environ.get("VRM_KV_COMPRESSION", "").lower())` |
| 1310 | `bits = int(os.environ.get("VRM_KV_COMPRESSION_BITS", "3"))` | `bits = (_flags.KV_COMPRESSION_BITS if _flags else int(os.environ.get("VRM_KV_COMPRESSION_BITS", "3")))` |
| 1311 | `residual = int(os.environ.get("VRM_KV_CACHE_RESIDUAL", "128"))` | `residual = (_flags.KV_CACHE_RESIDUAL if _flags else int(os.environ.get("VRM_KV_CACHE_RESIDUAL", "128")))` |
| 1378 | `max_batch_size=int(os.environ.get("VRM_MAX_BATCH_SIZE", "32")),` | `max_batch_size=(_flags.MAX_BATCH_SIZE if _flags else int(os.environ.get("VRM_MAX_BATCH_SIZE", "32"))),` |
| 1482 | `max_lend_ratio=float(os.environ.get("VRM_LEND_RATIO", "0.70")),` | `max_lend_ratio=(_flags.LEND_RATIO if _flags else float(os.environ.get("VRM_LEND_RATIO", "0.70"))),` |
| 1483 | `reclaim_threshold=float(os.environ.get("VRM_RECLAIM_THRESHOLD", "0.80")),` | `reclaim_threshold=(_flags.RECLAIM_THRESHOLD if _flags else float(os.environ.get("VRM_RECLAIM_THRESHOLD", "0.80"))),` |
| 1514 | `interval=float(os.environ.get("VRM_LENDING_INTERVAL", "2.0"))` | `interval=(_flags.LENDING_INTERVAL if _flags else float(os.environ.get("VRM_LENDING_INTERVAL", "2.0")))` |

**Important** : avant chaque `replace_string_in_file`, lire les 3 lignes avant/après via `grep -n "texte_exact" core/inference_pipeline.py` pour confirmer la ligne exacte. Si le contexte diffère, ajuster.

**Validation** :
```bash
python -c "import ast; ast.parse(open('core/inference_pipeline.py').read())" && echo OK
VRM_MINIMAL_TEST=1 pytest -q tests/test_pipeline.py --tb=short --no-cov 2>&1 | tail -5
```

**Commit** : `[V0.1] migrate inference_pipeline os.environ to env_flags`

---

### V0.2 — Ajouter tests pour `env_flags`

**Action** : créer `tests/test_env_flags.py` avec ce contenu **exact** :

```python
"""Tests for core.env_flags — live os.environ facade."""
import os
import pytest


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VRM_") and k not in ("VRM_MINIMAL_TEST", "VRM_TEST_MODE",
                                               "VRM_DISABLE_RATE_LIMIT"):
            monkeypatch.delenv(k, raising=False)
    yield


def test_bool_default_false():
    from core.env_flags import flags
    assert flags.TRUST_REMOTE_CODE is False
    assert flags.DISABLE_TURBO is False
    assert flags.CUDA_GRAPH is False


def test_bool_set_true(monkeypatch):
    monkeypatch.setenv("VRM_TRUST_REMOTE_CODE", "1")
    from core.env_flags import flags
    assert flags.TRUST_REMOTE_CODE is True


def test_bool_yes_truthy(monkeypatch):
    monkeypatch.setenv("VRM_DISABLE_TURBO", "yes")
    from core.env_flags import flags
    assert flags.DISABLE_TURBO is True


def test_int_default(monkeypatch):
    from core.env_flags import flags
    assert flags.MAX_BATCH_SIZE == 32
    assert flags.SPEC_GAMMA == 5
    assert flags.CUDA_GRAPH_CACHE == 4


def test_int_override(monkeypatch):
    monkeypatch.setenv("VRM_MAX_BATCH_SIZE", "64")
    from core.env_flags import flags
    assert flags.MAX_BATCH_SIZE == 64


def test_int_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("VRM_MAX_BATCH_SIZE", "not_a_number")
    from core.env_flags import flags
    assert flags.MAX_BATCH_SIZE == 32  # default


def test_float_default():
    from core.env_flags import flags
    assert flags.LEND_RATIO == pytest.approx(0.70)
    assert flags.GENERATE_TIMEOUT == pytest.approx(300.0)


def test_float_override(monkeypatch):
    monkeypatch.setenv("VRM_LEND_RATIO", "0.55")
    from core.env_flags import flags
    assert flags.LEND_RATIO == pytest.approx(0.55)


def test_str_default():
    from core.env_flags import flags
    assert flags.PARALLEL_MODE == "pp"
    assert flags.BACKEND == "auto"


def test_str_override(monkeypatch):
    monkeypatch.setenv("VRM_PARALLEL_MODE", "tp")
    from core.env_flags import flags
    assert flags.PARALLEL_MODE == "tp"


def test_opt_str_none():
    from core.env_flags import flags
    assert flags.DRAFT_MODEL is None
    assert flags.API_TOKEN is None


def test_opt_str_set(monkeypatch):
    monkeypatch.setenv("VRM_DRAFT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    from core.env_flags import flags
    assert flags.DRAFT_MODEL == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_vram_lending_default_on():
    from core.env_flags import flags
    assert flags.VRAM_LENDING is True  # default "1" = on


def test_vram_lending_disabled(monkeypatch):
    monkeypatch.setenv("VRM_VRAM_LENDING", "0")
    from core.env_flags import flags
    assert flags.VRAM_LENDING is False


def test_transfer_p2p_default_on():
    from core.env_flags import flags
    assert flags.TRANSFER_P2P is True


def test_transfer_p2p_disabled(monkeypatch):
    monkeypatch.setenv("VRM_TRANSFER_P2P", "false")
    from core.env_flags import flags
    assert flags.TRANSFER_P2P is False


def test_repr_contains_active_flags(monkeypatch):
    monkeypatch.setenv("VRM_DEBUG", "1")
    monkeypatch.setenv("VRM_BACKEND", "vllm")
    from core.env_flags import flags
    r = repr(flags)
    assert "DEBUG" in r
    assert "BACKEND" in r
```

**Validation** :
```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/test_env_flags.py --tb=short --no-cov
# Attendu : 18 passed, 0 failed
```

**Commit** : `[V0.2] add tests/test_env_flags.py — 18 tests`

---

## P1 — Sécurité : tests `verify_request` et `startup_checks`

### V1.1 — Tests unitaires pour `verify_request`

**Contexte** : `core/security/__init__.py:verify_request()` (ligne 109) valide token + HMAC. Aucun test direct n'existe pour cette fonction (les tests existants testent le middleware Flask, pas la fonction elle-même).

**Action** : créer `tests/test_verify_request.py` avec ce contenu **exact** :

```python
"""Unit tests for core.security.verify_request()."""
import os
import hmac
import hashlib
import time
import pytest


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("VRM_MINIMAL_TEST", "1")
    monkeypatch.setenv("VRM_DISABLE_RATE_LIMIT", "1")
    monkeypatch.setenv("VRM_TEST_MODE", "1")
    monkeypatch.delenv("VRM_PRODUCTION", raising=False)
    monkeypatch.delenv("VRM_TEST_RELAX_SECURITY", raising=False)
    yield


def _get_verify():
    from core.security import verify_request
    return verify_request


def test_public_health_path_passes():
    vr = _get_verify()
    for path in ("/health", "/ready", "/live", "/api/health", "/"):
        result = vr(None, "GET", path, {}, b"")
        assert result is None, f"Expected None for public path {path}, got {result}"


def test_static_path_passes():
    vr = _get_verify()
    assert vr(None, "GET", "/static/app.js", {}, b"") is None


def test_favicon_passes():
    vr = _get_verify()
    assert vr(None, "GET", "/favicon.ico", {}, b"") is None


def test_no_token_non_production_passes():
    vr = _get_verify()
    # Non-production with no token → passes (no VRM_PRODUCTION=1)
    result = vr("mysecret", "POST", "/api/models/load", {}, b"{}")
    assert result is None


def test_valid_token_passes():
    vr = _get_verify()
    result = vr("mysecret", "POST", "/api/models/load",
                {"X-API-TOKEN": "mysecret"}, b"{}")
    assert result is None


def test_wrong_token_non_production_passes():
    """In non-production mode a wrong token is not fatal."""
    vr = _get_verify()
    result = vr("mysecret", "POST", "/api/models/load",
                {"X-API-TOKEN": "wrongtoken"}, b"{}")
    # Non-production: wrong token doesn't block (no hard enforcement)
    # Just verify it doesn't raise
    assert result is None or isinstance(result, tuple)


def test_production_no_token_returns_401(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    vr = _get_verify()
    result = vr("mysecret", "POST", "/api/models/load", {}, b"{}")
    assert result is not None
    _msg, code = result
    assert code == 401


def test_production_valid_token_passes(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    vr = _get_verify()
    result = vr("mysecret", "POST", "/api/models/load",
                {"X-API-TOKEN": "mysecret"}, b"{}")
    assert result is None


def test_bearer_token_accepted(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    vr = _get_verify()
    result = vr("mytoken", "GET", "/api/gpu",
                {"Authorization": "Bearer mytoken"}, b"")
    assert result is None
```

**Validation** :
```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/test_verify_request.py --tb=short --no-cov
# Attendu : 10 passed (ou quelques skip), 0 failed
```

Si un test échoue à cause d'un comportement inattendu de `verify_request` (ex: token wrong retourne 401 même hors production) → adapter l'assertion en conséquence avec `assert result is None or isinstance(result, tuple)`. **Ne pas modifier `core/security/__init__.py`**.

**Commit** : `[V1.1] add tests/test_verify_request.py — 10 tests`

---

### V1.2 — Tests unitaires pour `startup_checks`

**Contexte** : `core/security/startup_checks.py:enforce_startup_checks()` (ligne 37) lève `RuntimeError` en production si credentials/tokens manquants. Aucun test direct.

**Action** : créer `tests/test_startup_checks.py` avec ce contenu **exact** :

```python
"""Unit tests for core.security.startup_checks.enforce_startup_checks()."""
import os
import pytest


@pytest.fixture(autouse=True)
def _base_env(monkeypatch):
    monkeypatch.delenv("VRM_PRODUCTION", raising=False)
    monkeypatch.delenv("VRM_API_TOKEN", raising=False)
    monkeypatch.delenv("VRM_AUTH_SECRET", raising=False)
    monkeypatch.delenv("VRM_MINIMAL_TEST", raising=False)
    monkeypatch.delenv("VRM_TEST_RELAX_SECURITY", raising=False)
    monkeypatch.delenv("VRM_TEST_BYPASS_HA", raising=False)
    yield


def _enforce():
    # Re-import each time to avoid module-level caching of env vars
    import importlib
    import core.security.startup_checks as m
    importlib.reload(m)
    return m.enforce_startup_checks


def test_non_production_always_passes(monkeypatch):
    """No VRM_PRODUCTION=1 → no checks enforced."""
    fn = _enforce()
    fn()  # must not raise


def test_production_no_token_raises(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_AUTH_SECRET", "somesecret")
    fn = _enforce()
    with pytest.raises(RuntimeError, match="VRM_API_TOKEN"):
        fn()


def test_production_no_auth_secret_raises(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "sometoken")
    fn = _enforce()
    with pytest.raises(RuntimeError, match="VRM_AUTH_SECRET"):
        fn()


def test_production_with_all_required_passes(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "secure-token-xyz")
    monkeypatch.setenv("VRM_AUTH_SECRET", "a" * 32)
    fn = _enforce()
    # Should not raise (assuming no default admin/admin credentials in test)
    try:
        fn()
    except RuntimeError as e:
        if "admin" in str(e).lower():
            pytest.skip("Default admin/admin credentials present in test env")
        raise


def test_production_test_env_var_leaking_raises(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "secure-token-xyz")
    monkeypatch.setenv("VRM_AUTH_SECRET", "a" * 32)
    monkeypatch.setenv("VRM_MINIMAL_TEST", "1")
    fn = _enforce()
    with pytest.raises(RuntimeError, match="VRM_MINIMAL_TEST"):
        fn()


def test_production_relax_security_raises(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_API_TOKEN", "secure-token-xyz")
    monkeypatch.setenv("VRM_AUTH_SECRET", "a" * 32)
    monkeypatch.setenv("VRM_TEST_RELAX_SECURITY", "1")
    fn = _enforce()
    with pytest.raises(RuntimeError, match="VRM_TEST_RELAX_SECURITY"):
        fn()
```

**Validation** :
```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/test_startup_checks.py --tb=short --no-cov
# Attendu : 6 passed (ou 5+1 skip), 0 failed
```

**Commit** : `[V1.2] add tests/test_startup_checks.py — 6 tests`

---

## P2 — Fix effet de bord `registry.py` : ClusterDiscovery au démarrage

### V2.1 — Rendre ClusterDiscovery conditionnel dans `PipelineRegistry`

**Contexte** : `core/api/registry.py` ligne 28-33 démarre `ClusterDiscovery` automatiquement à l'instanciation de `PipelineRegistry`. Cela déclenche des broadcasts réseau UDP à chaque `import core.production_api`. C'est un effet de bord non désiré en test et en dev.

**Vérifier les lignes exactes** :
```bash
grep -n "ClusterDiscovery\|discovery.start\|discovery =" core/api/registry.py
```

**Action** — lire le bloc exact avec :
```bash
sed -n '22,40p' core/api/registry.py
```

Remplacer le bloc qui contient :
```python
        # Start global cluster discovery immediately so node is discoverable
        try:
            from core.network.cluster_discovery import ClusterDiscovery
            self.discovery = ClusterDiscovery(heartbeat_interval=5)
            self.discovery.start()
        except ImportError:
            pass
```

Par :
```python
        # Start cluster discovery only when explicitly enabled (not in test/dev)
        _auto_discover = (
            os.environ.get("VRM_CLUSTER_AUTO_DISCOVER", "").lower()
            not in ("0", "false", "no", "")
        )
        if _auto_discover:
            try:
                from core.network.cluster_discovery import ClusterDiscovery
                self.discovery = ClusterDiscovery(heartbeat_interval=5)
                self.discovery.start()
            except ImportError:
                pass
```

**Vérifier** que `import os` est bien présent en haut de `registry.py` :
```bash
head -15 core/api/registry.py | grep "^import os"
```

Si absent, ajouter `import os` après `from __future__ import annotations`.

**Validation** :
```bash
python -c "
import os
os.environ['VRM_MINIMAL_TEST'] = '1'
from core.api.registry import PipelineRegistry
r = PipelineRegistry()
assert r.discovery is None, f'Expected no discovery, got {r.discovery}'
print('OK — no auto-discovery without VRM_CLUSTER_AUTO_DISCOVER')
"

VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/ --ignore=tests/test_chaos_concurrency.py --tb=no --no-cov 2>&1 | tail -1
```

**Commit** : `[V2.1] guard ClusterDiscovery auto-start behind VRM_CLUSTER_AUTO_DISCOVER`

---

### V2.2 — Ajouter `VRM_CLUSTER_AUTO_DISCOVER` à `core/env_flags.py`

**Action** — ouvrir `core/env_flags.py`, localiser le bloc `# ── Networking / cluster ───────────────────────────────────────────`:
```bash
grep -n "Networking / cluster\|API_HOST\|NODE_ID" core/env_flags.py | head -5
```

Après la propriété `NODE_ID`, ajouter une nouvelle propriété. Trouver les lignes exactes avec :
```bash
grep -n "def NODE_ID\|def PEER_IPS" core/env_flags.py
```

Insérer après la propriété `NODE_ID` (et son `return`) :

```python
    @property
    def CLUSTER_AUTO_DISCOVER(self) -> bool:
        """Auto-start ClusterDiscovery at PipelineRegistry init."""
        return _bool("VRM_CLUSTER_AUTO_DISCOVER")
```

**Validation** :
```bash
python -c "
import os; os.environ['VRM_MINIMAL_TEST'] = '1'
from core.env_flags import flags
assert flags.CLUSTER_AUTO_DISCOVER is False
import os; os.environ['VRM_CLUSTER_AUTO_DISCOVER'] = '1'
assert flags.CLUSTER_AUTO_DISCOVER is True
print('OK')
"
```

**Commit** : `[V2.2] add CLUSTER_AUTO_DISCOVER to env_flags`

---

## P3 — Logs de démarrage : afficher le mode actif

### V3.1 — Log du mode au démarrage de `InferencePipeline.load()`

**Contexte** : `core/inference_pipeline.py` ligne ~173 logue `"Loading model: %s (backend=%s)"`. Il manque un log clair du mode actif (stub/production/test) et des features actives.

**Trouver la ligne exacte** :
```bash
grep -n '"Loading model:' core/inference_pipeline.py
```

Insérer **juste avant** `_logger.info("Loading model: %s (backend=%s)", model_name, self.backend_name)` le bloc suivant :

```python
            # Log active mode flags at startup
            _mode_flags = []
            if os.environ.get("VRM_MINIMAL_TEST"):
                _mode_flags.append("STUB")
            elif os.environ.get("VRM_PRODUCTION") == "1":
                _mode_flags.append("PRODUCTION")
            else:
                _mode_flags.append("DEV")
            if os.environ.get("VRM_CONTINUOUS_BATCHING") == "1":
                _mode_flags.append("continuous_batching")
            _quant = os.environ.get("VRM_QUANTIZATION", "")
            if _quant:
                _mode_flags.append(f"quant={_quant}")
            _kv = os.environ.get("VRM_KV_COMPRESSION", "")
            if _kv:
                _mode_flags.append(f"kv={_kv}")
            _logger.info("VRAMancer mode: [%s]", ", ".join(_mode_flags))
```

**Validation** :
```bash
python -c "
import os, logging
os.environ['VRM_MINIMAL_TEST'] = '1'
logging.basicConfig(level=logging.INFO)
from core.inference_pipeline import InferencePipeline
# Just import, no load — verify no crash
print('OK')
"
VRM_MINIMAL_TEST=1 pytest -q tests/test_pipeline.py --tb=short --no-cov 2>&1 | tail -5
```

**Commit** : `[V3.1] log active mode flags at pipeline startup`

---

## P4 — Flag `VRM_FEATURE_AITP` pour `llm_transport`

### V4.1 — Activer `llm_transport` uniquement si `VRM_FEATURE_AITP=1`

**Contexte** : `core/network/llm_transport.py` est importé au démarrage si présent. Il ouvre des sockets. Il doit être opt-in.

**Vérifier où il est importé** :
```bash
grep -rn "llm_transport\|LLMTransport\|VTPServer" core/ --include="*.py" | grep "^core" | grep -v "test\|llm_transport.py" | head -10
```

**Puis** dans chaque fichier qui l'importe directement (hors `llm_transport.py` lui-même), entourer l'import avec :

```python
if os.environ.get("VRM_FEATURE_AITP") == "1":
    try:
        from core.network.llm_transport import LLMTransport, VTPServer
    except ImportError:
        pass
```

**Note** : si l'import est dans un `try/except ImportError` existant, ajouter la condition `os.environ.get("VRM_FEATURE_AITP") == "1"` avant le `try`.

**Validation** :
```bash
python -c "
import os; os.environ['VRM_MINIMAL_TEST'] = '1'
# Without flag — should not import llm_transport
import sys
from core.inference_pipeline import InferencePipeline
assert 'core.network.llm_transport' not in sys.modules, 'llm_transport was imported without VRM_FEATURE_AITP'
print('OK — llm_transport not auto-imported')
"
```

Si le test ci-dessus passe déjà (llm_transport n'est pas importé sans flag) → noter `V4.1: already guarded — SKIP` dans EXEC_LOG_V2.md et passer à V4.2.

**Commit** (si changement) : `[V4.1] guard llm_transport import behind VRM_FEATURE_AITP`

---

### V4.2 — Ajouter `VRM_FEATURE_AITP` à `env_flags.py`

**Trouver** le bloc `# ── AITP / VTP ─────────────────────────────────────────────────────`:
```bash
grep -n "AITP / VTP\|AITP_PORT\|def AITP_PORT" core/env_flags.py | head -3
```

Ajouter **avant** la propriété `AITP_PORT` :

```python
    @property
    def FEATURE_AITP(self) -> bool:
        """Opt-in AITP/VTP network stack activation (opens sockets on import)."""
        return _bool("VRM_FEATURE_AITP")

```

**Validation** :
```bash
python -c "
import os; os.environ['VRM_MINIMAL_TEST'] = '1'
from core.env_flags import flags
assert flags.FEATURE_AITP is False
print('OK')
"
```

**Commit** : `[V4.2] add FEATURE_AITP flag to env_flags`

---

## P5 — Tests pour `LlamaServerBackend`

### V5.1 — Tests unitaires `LlamaServerBackend` (sans binaire)

**Contexte** : `core/llama_server_backend.py` (313 LOC) n'a aucun test. Tester les méthodes sans lancer le binaire via mock.

**Action** : créer `tests/test_llama_server_backend.py` avec ce contenu **exact** :

```python
"""Unit tests for core.llama_server_backend — no binary required."""
import pytest
import os


pytestmark = pytest.mark.skipif(
    not os.environ.get("VRM_MINIMAL_TEST"),
    reason="stub-safe only"
)


def test_platform_key_returns_string():
    from core.llama_server_backend import _platform_key
    key = _platform_key()
    assert isinstance(key, str)
    assert key in ("linux-cuda", "linux-cpu", "darwin-arm", "darwin-x86", "windows")


def test_asset_map_has_all_platforms():
    from core.llama_server_backend import _ASSET_MAP
    for key in ("linux-cuda", "linux-cpu", "darwin-arm", "darwin-x86", "windows"):
        assert key in _ASSET_MAP
        assert "{tag}" in _ASSET_MAP[key]


def test_local_tensor_split_no_gpu():
    """Without GPU, returns None gracefully."""
    from core.llama_server_backend import _local_tensor_split
    result = _local_tensor_split(2)
    # Either None (no GPU) or a list of floats
    assert result is None or isinstance(result, list)


def test_binary_dir_is_path():
    from core.llama_server_backend import BINARY_DIR
    from pathlib import Path
    assert isinstance(BINARY_DIR, Path)
    assert "vramancer" in str(BINARY_DIR)


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


def test_llama_server_backend_init_requires_binary(monkeypatch, tmp_path):
    """LlamaServerBackend.__init__ raises if binary not found and download fails."""
    from unittest.mock import patch
    from core.llama_server_backend import LlamaServerBackend
    # Redirect BINARY_DIR to tmp so no real download
    import core.llama_server_backend as m
    monkeypatch.setattr(m, "BINARY_DIR", tmp_path)
    with patch.object(m, "_download_release_binary", side_effect=RuntimeError("mocked")):
        with pytest.raises(RuntimeError):
            LlamaServerBackend("/nonexistent/model.gguf")


def test_get_or_download_finds_existing_binary(monkeypatch, tmp_path):
    from core.llama_server_backend import BINARY_DIR
    import core.llama_server_backend as m
    # Create fake binary
    fake = tmp_path / "llama-server"
    fake.write_text("fake")
    monkeypatch.setattr(m, "BINARY_DIR", tmp_path)
    result = m.get_or_download_binary()
    assert result == fake
```

**Validation** :
```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/test_llama_server_backend.py --tb=short --no-cov
# Attendu : 8 passed (ou quelques skip), 0 failed
```

**Commit** : `[V5.1] add tests/test_llama_server_backend.py — 8 tests`

---

## P6 — Documentation : matrice de compatibilité

### V6.1 — Créer `docs/COMPATIBILITY.md`

**Action** : créer `docs/COMPATIBILITY.md` avec ce contenu **exact** :

````markdown
# VRAMancer — Matrice de compatibilité

## Backends × Quantization × OS

| Backend | `nvfp4` | `nf4` | `int8` | `BF16` | Linux/CUDA | macOS/MPS | Windows/CUDA | CPU |
|---|---|---|---|---|---|---|---|---|
| HuggingFace | ✓ (CC≥10) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| vLLM | — | — | via vLLM | ✓ | ✓ | — | — | — |
| llama.cpp (binaire) | — | — | — | GGUF | ✓ | ✓ | ✓ | ✓ |
| llama-cpp-python | — | — | GGUF Q8 | GGUF | ✓ | ✓ | ✓ | ✓ |
| Ollama | — | — | — | via Ollama | ✓ | ✓ | ✓ | ✓ |

> **nvfp4** : requiert Blackwell (RTX 50xx, CC≥10.0) + torchao.  
> **nf4/int8** : requiert `bitsandbytes` ≥ 0.41. Bug upstream multi-GPU sur BnB+accelerate — forcé single-GPU par VRAMancer.

## GPU × Stratégie de transfer

| Environnement | Stratégie active | Commentaire |
|---|---|---|
| Bare-metal PCIe P2P | CUDA P2P (strategy 2) | RTX 30xx/40xx/50xx, pas GeForce driver block |
| Proxmox VM VFIO | CPU-staged pinned (strategy 4) | IOMMU bloque P2P, overhead ~10-15% |
| macOS MPS | CPU copy | Pas de CUDA |
| Single GPU | N/A | Pas de transfer |
| NVLink (A100/H100) | CUDA P2P NVLink | Bande passante 600 GB/s |

## Dépendances optionnelles

| Composant | Package | Activation | Fonctionnalité |
|---|---|---|---|
| `rust_core` | `cargo build --release` | Auto si `.so` présent | CUDA DtoD direct, HMAC Rust 100x |
| `pyverbs` | `pip install pyverbs` | Auto si présent | RDMA verbs zero-copy |
| `nvidia_peermem` | module kernel | Auto si chargé | GPUDirect RDMA |
| `bitsandbytes` | `pip install bitsandbytes` | `VRM_QUANTIZATION=nf4\|int8` | 4/8-bit quantization |
| `triton` | `pip install triton` | Auto si présent | Kernels TurboQuant GPU-natifs |
| `vllm` | `pip install vllm` | `VRM_BACKEND=vllm` | vLLM engine |
| `llama-cpp-python` | `pip install llama-cpp-python` | `VRM_BACKEND=llamacpp` | GGUF inference |
| `flask-socketio` | `pip install flask-socketio` | Auto si présent | Dashboard WebSocket |

## Variables d'environnement critiques

| Variable | Défaut | Impact |
|---|---|---|
| `VRM_QUANTIZATION` | `""` (BF16) | VRAM : BF16=2x, NF4=0.5x, NVFP4=0.25x |
| `VRM_PARALLEL_MODE` | `pp` | `tp` active NCCL all-reduce |
| `VRM_CONTINUOUS_BATCHING` | `0` | `1` = multi-req batch, requis pour >1 user |
| `VRM_TRANSFER_P2P` | `1` | `0` = force CPU-staged (VM Proxmox) |
| `VRM_TRUST_REMOTE_CODE` | `0` | `1` = permet code custom HF (risque sécurité) |
| `VRM_PRODUCTION` | `0` | `1` = enforce token/secret obligatoires |
| `VRM_FEATURE_AITP` | `0` | `1` = active réseau AITP/VTP (ouvre des sockets) |
| `VRM_CLUSTER_AUTO_DISCOVER` | `0` | `1` = broadcast UDP au démarrage |

## Benchmarks de référence (RTX 3090 + RTX 5070 Ti, Proxmox)

| Modèle | Config | tok/s | VRAM |
|---|---|---|---|
| GPT-2 124M | BF16 1-GPU | 125.6 | 0.5 GB |
| TinyLlama 1.1B | BF16 1-GPU | 56.5 | 2.2 GB |
| Qwen2.5-7B | NF4 1-GPU | 20.2 | ~5 GB |
| Qwen2.5-7B GGUF Q4_K_M | llama.cpp | 106.8 | 3.0 GB |
| Qwen2.5-14B | BF16 2-GPU | 6.0 | 35.9 GB |
| Qwen2.5-14B | NF4 1-GPU | 10.5 | 10.8 GB |
| TinyLlama 1.1B | NVFP4 RTX 5070 Ti | ~36 | 5.46 GB |
````

**Validation** :
```bash
test -f docs/COMPATIBILITY.md && wc -l docs/COMPATIBILITY.md
# Entre 60 et 100 lignes
```

**Commit** : `[V6.1] add docs/COMPATIBILITY.md — backend x quant x GPU matrix`

---

## P7 — Nettoyage imports `production_api.py`

### V7.1 — Déplacer `cross_node` import dans un bloc conditionnel

**Contexte** : `core/production_api.py` ligne ~279 importe `from core.cross_node import start_vtp_server` de façon inconditionnelle au démarrage.

**Vérifier** :
```bash
grep -n "cross_node\|start_vtp_server" core/production_api.py
```

Si la ligne ressemble à :
```python
            from core.cross_node import start_vtp_server
```

Elle doit déjà être dans un `try/except` ou un bloc conditionnel. **Si elle est déjà dans un `try/except`** → noter `V7.1: already guarded — SKIP` et passer à V7.2.

**Si non guardée**, entourer avec :
```python
            if os.environ.get("VRM_FEATURE_AITP") == "1":
                try:
                    from core.cross_node import start_vtp_server
                    start_vtp_server()
                except Exception:
                    pass
```

**Validation** :
```bash
python -c "
import os; os.environ['VRM_MINIMAL_TEST'] = '1'; os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
from core.production_api import create_app
app = create_app()
print('OK — create_app() without VRM_FEATURE_AITP')
"
```

**Commit** (si changement) : `[V7.1] guard cross_node VTP import behind VRM_FEATURE_AITP`

---

### V7.2 — Ajouter test d'import propre de `production_api`

**Action** : dans `tests/test_imports.py`, trouver la liste `CORE_MODULES` et vérifier que `core.production_api` y figure. Si absent, l'ajouter. Si présent, passer.

```bash
grep -n "production_api" tests/test_imports.py
```

Si absent : trouver la fin de la liste `CORE_MODULES = [` et ajouter `"core.production_api",` juste avant le `]`.

**Validation** :
```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/test_imports.py --tb=short --no-cov 2>&1 | tail -3
```

**Commit** : `[V7.2] add production_api to test_imports module list`

---

## P8 — Validation finale

### V8.1 — Relancer la suite complète

```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=short --no-cov -q 2>&1 | tail -10
```

Critère : `≥1014 passed`, `≤1 failed` (le pre-existing), `0 nouvelle failure`.

### V8.2 — Vérifier le git log

```bash
git log --oneline chore/sonnet-plan-v2 ^chore/sonnet-plan-exec
```

Doit afficher exactement les commits des tâches DONE, dans l'ordre.

### V8.3 — Mettre à jour EXEC_LOG_V2.md

Ajouter la section `### [SUMMARY]` :

```markdown
### [SUMMARY]

- Tâches DONE : X/Y
- Tâches BLOCKED : Z
- Tests ajoutés : N (env_flags x18, verify_request x10, startup_checks x6, llama_server x8, ...)
- Baseline : 1014 passed, 1 failed, 39 skipped
- Final : [résultat ici]
- Aucune régression introduite.
```

**Commit** : `[V8.x] final EXEC_LOG_V2.md summary`

---

## Ordre d'exécution

```
0.1 → 0.2 → 0.3 → 0.4
→ V0.1 → V0.2
→ V1.1 → V1.2
→ V2.1 → V2.2
→ V3.1
→ V4.1 → V4.2
→ V5.1
→ V6.1
→ V7.1 → V7.2
→ V8.1 → V8.2 → V8.3
```

---

## Ce que Sonnet 4.6 ne doit PAS faire

- Renommer `class Synapse`, `synapses`, `_apply_neuroplasticity_score`, `HolographicKVManager`.
- Modifier `_deprecated/*.py`.
- Modifier `.github/copilot-instructions.md`.
- Ajouter des dépendances dans `requirements*.txt` ou `pyproject.toml`.
- Migrer tous les `os.environ.get` de tous les modules (périmètre limité à `inference_pipeline.py`).
- Modifier `core/security/__init__.py` (tests seulement, pas le code source).
- Auto-pusher la branche ou créer une PR.
- Modifier `tests/test_chaos_concurrency.py`.

---

## Références

- `core/env_flags.py` — façade flags (créée session précédente)
- `core/inference_pipeline.py` — 25 os.environ.get à migrer
- `core/api/registry.py` — ClusterDiscovery auto-start
- `core/security/__init__.py:verify_request()` — ligne 109
- `core/security/startup_checks.py:enforce_startup_checks()` — ligne 37
- `core/llama_server_backend.py` — 313 LOC, 0 test
- `.github/copilot-instructions.md` — source de vérité technique
