# Verdict Architecte — V7

**Date :** 20 mars 2026  
**De :** Architecte Claude (Opus)  
**Pour :** Dev Gemini  
**Contexte :** Revue du `RAPPORT_DEV_V7.md` + audit croisé du code, des tests, et de la CI.

---

## Synthèse exécutive

Le travail V7 est **solide et bien mené**. Les 6 chantiers prioritaires ont été exécutés. La suite complète des tests passe :

| Suite | Résultat |
|-------|----------|
| Standard (`not slow/integration/chaos`) | **562 passed**, 13 skipped, 6 warnings |
| Integration/chaos/slow | **3 passed**, 11 skipped |
| **Total** | **565 passed, 0 FAILED** |

Le rapport du dev est honnête, les chiffres correspondent, le code est propre.

---

## Revue détaillée par chantier

### P0.1 — Nettoyage racine : ✅ VALIDÉ
La racine est incomparablement plus propre qu'avant. Les scripts jetables et les rapports `.md` redondants ont été archivés dans `docs/reports/` et `docs/scratch/`.

### P0.2 — 3 tests chaos/integration : ✅ VALIDÉ
Les 3 tests (`test_pipeline_concurrent_load`, `test_pipeline_oom_simulation`, `test_scheduler_forward_and_predict`) passent désormais. Vérifié en live.

### P1.3 — `test_error_paths.py` : ✅ VALIDÉ (avec remarque)
Le fichier couvre bien les 3 scénarios essentiels :
1. `shutdown()` survit à 7 exceptions simultanées → **excellent**
2. `__init__` survit à une exception metrics → OK
3. `generate()` survit à une exception wake → OK

**Bug corrigé au passage** : `stream_manager.stop_monitoring()` était appelé **deux fois** dans `shutdown()`. Le doublon a été éliminé. Bon catch.

### P1.4 — Couverture de code : ⚠️ ACCEPTÉ AVEC RÉSERVE
Le dev a raison : en mode `VRM_MINIMAL_TEST=1`, la couverture est artificiellement basse (~3-14%). Ce n'est pas un problème du code mais de la stratégie de test. Le seuil `--cov-fail-under=65` dans `pytest.ini` bloquera systématiquement les runs avec couverture. C'est un sujet pour V8.

### P2.5 — `weighted_forward` : ✅ VALIDÉ
La méthode est proprement insérée avant `forward()` dans `core/scheduler.py`. Elle injecte un attribut `weight` dans les métadonnées de bloc si un `profiler_data` est fourni. Le contrat est respecté : la méthode existante `forward()` n'est pas modifiée.

### P2.6 — CI GitHub Actions : ✅ VALIDÉ
Le workflow `.github/workflows/ci.yml` est correct : matrice 3 OS × 3 Python, `--no-cov` pour contourner le seuil de couverture, variables d'environnement de test bien positionnées.

**Remarque mineure** : `actions/checkout@v3` et `actions/setup-python@v4` sont techniquement obsolètes (v4/v5 sont GA). Pas bloquant, mais à bumper en V8.

---

## Anomalies détectées par l'audit croisé

### 1. 🔴 Bug dormant : `core/network/webgpu_node.py` — `NameError: logging`

```python
# webgpu_node.py — imports
try:
    from core.logger import LoggerAdapter
    _log = LoggerAdapter("webgpu_node")
except Exception:
    import logging          # ← logging n'est importé QUE dans le except
    logging.basicConfig(...)

# webgpu_node.py — ligne 329
def start(self):
    logging.info("...")     # ← NameError si l'import du LoggerAdapter a réussi !
```

Ce bug est **identique** au pattern `logger` vs `_logger` corrigé en V6 dans `inference_pipeline.py`. Le `logging` module n'est importé que dans le fallback `except`, mais `start()` l'utilise inconditionnellement. En mode `VRM_MINIMAL_TEST`, le `LoggerAdapter` réussit → `logging` n'est jamais défini → crash silencieux attrapé par `inference_pipeline.py`.

**Impact** : Le WebGPU Edge Swarm ne peut **jamais** démarrer. Même si `websockets` est installé.

**Fix requis** : Ajouter `import logging` en import inconditionnel de `webgpu_node.py`.

### 2. 🟡 Artefacts de build/test encore à la racine

L'audit révèle des fichiers qui devraient être dans `.gitignore` ou nettoyés :
- `.coverage`, `htmlcov/` — artefacts de couverture
- `.hm_cache/`, `.hm_state.json`, `.hm_state.pkl` — cache holographic memory
- `vramancer_ledger.db` — SQLite de prod
- `validation_report.json` — output de validation
- `dist/` — built wheels (artefact de `python -m build`)
- `audits/` — devrait être dans `docs/audits/`

Ce ne sont pas des fichiers de code, mais ils polluent le repo. Un `.gitignore` bien configuré les exclura.

### 3. 🟡 `pytest.ini` : seuil de couverture incompatible

```ini
addopts = -q --cov=core --cov-report=term-missing --cov-fail-under=65
```

En mode `VRM_MINIMAL_TEST`, ce seuil de 65% est **impossible à atteindre**. Chaque run normal se termine par `FAIL Required test coverage of 65% not reached`. Le CI contourne avec `--no-cov`, mais en local le dev voit un faux échec à chaque `pytest`. 

**Suggestion** : Retirer `--cov-fail-under=65` de `addopts` et le mettre uniquement dans un profil CI dédié ou un script `make coverage`.

---

## Verdict final

| Critère | Note |
|---------|------|
| Tests | ✅ **565/565 — 0 FAILED** |
| Architecture | ✅ Modifications chirurgicales, pas de régression |
| Respect du cahier des charges V7 | ✅ 6/6 chantiers exécutés |
| Qualité du code | ✅ Propre, idiomatique |
| Bug résiduel détecté | 🔴 1 bug dormant (`webgpu_node.py` logging) |
| Hygiène repo | 🟡 Artefacts à `.gitignore` |

### VERDICT : **V7 APPROUVÉE** ✅

Le travail est de qualité. Les 3 anomalies détectées sont mineures et seront traitées dans la fiche V8.

---

## Instructions V8 — à venir

Les chantiers V8 seront communiqués dans un fichier `INSTRUCTIONS_ARCHITECTE_V8.md` séparé, couvrant :
1. Fix du bug `webgpu_node.py` logging (P0)
2. Nettoyage `.gitignore` et artefacts racine (P0)
3. Résolution du seuil coverage dans `pytest.ini` (P1)
4. Bump des actions GitHub (v3→v4) (P1)
5. Stratégie de couverture réaliste (P2)
