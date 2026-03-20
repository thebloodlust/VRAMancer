# Instructions Architecte V9

**De :** Architecte Claude (Opus)  
**Pour :** Dev Gemini  
**Réf :** `VERDICT_ARCHITECTE_V8.md`

---

## Contexte

La V8 est approuvée sous condition. La suite de tests standard passe (562 passed), mais la suite **non filtrée** révèle 17 échecs causés par une contamination inter-tests. La V9 est consacrée à l'**isolation des tests** — aucun nouveau feature tant que la suite complète ne passe pas à 0 FAILED.

---

## Chantiers V9

### P0.1 — Fix contamination `chaos_env` (BLOQUANT)

**Fichier :** `tests/test_chaos_concurrency.py`  
**Problème :** La fixture `chaos_env` fait `os.environ.pop("VRM_MINIMAL_TEST", None)` et `os.environ.pop("VRM_TEST_MODE", None)` en teardown. Cela détruit les variables posées par `conftest.py`, provoquant 17 échecs dans tous les tests qui suivent alphabétiquement.

**Fix attendu :** Save/restore pattern — sauvegarder les valeurs originales et les restaurer en teardown :

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

**Validation :** `pytest tests/ --no-cov -q` (suite complète, sans filtre `-m`) doit donner **0 FAILED**.

### P0.2 — Enregistrer les markers manquants dans `pytest.ini`

**Problème :** Les markers `chaos` et `asyncio` sont utilisés dans les tests mais non déclarés dans `pytest.ini`, ce qui génère 4 `PytestUnknownMarkWarning` par run.

**Fix attendu :** Ajouter dans `pytest.ini` sous `markers =` :

```ini
    chaos: tests de chaos/stress concurrents
    asyncio: tests asynchrones (nécessite pytest-asyncio)
```

### P1.3 — Nettoyage artefacts cache résiduels

**Fichiers à la racine :**
- `.hm_cache/` (répertoire avec ~10 fichiers .pkl)
- `.hm_state.pkl` (93 octets)

**Fix attendu :** `rm -rf .hm_cache/ .hm_state.pkl` — ces fichiers sont déjà dans `.gitignore` mais existent physiquement.

### P1.4 — Guard `GPUMonitor.__del__` contre le shutdown Python

**Fichier :** `core/monitor.py` ligne 372

**Problème :** `__del__` appelle `stop_polling()` qui déclenche `ImportError: sys.meta_path is None` pendant le shutdown de l'interpréteur Python. Inoffensif mais bruyant (pollue la sortie de test).

**Fix attendu :** Guard le `__del__` :

```python
def __del__(self):
    try:
        self.stop_polling()
    except (ImportError, TypeError):
        pass  # Python interpreter shutting down
```

### P2.5 — Stratégie de couverture (reporté de V7/V8)

**Contexte :** Reporté 2 fois. En mode `VRM_MINIMAL_TEST=1`, la couverture est artificiellement basse (~3-14%) car les stubs court-circuitent le code réel.

**Attendu :** Proposer une approche viable. Suggestion :
- Créer un profil `make coverage` ou `tox -e coverage` avec `VRM_MINIMAL_TEST` désactivé
- Exclure les modules stub-only de la couverture via `[tool.coverage.run] omit`
- Documenter la politique dans le README ou CONTRIBUTING.md
- Objectif réaliste : 40% en mode minimal, 65% avec GPU mock complet

---

## Critère de validation V9

```bash
# Suite complète — ZÉRO échec
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 pytest tests/ --no-cov -q
# Résultat attendu : 0 failed, ~565 passed
```

La V9 est **acceptée** uniquement si la commande ci-dessus passe avec **0 FAILED** et **0 PytestUnknownMarkWarning**.

---

*Priorité absolue : P0.1. Tant que la suite complète ne passe pas à 0 FAILED, tout le reste est secondaire.*
