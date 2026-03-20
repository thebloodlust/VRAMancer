# Verdict Architecte — V10

**Date :** 20 mars 2026  
**De :** Architecte Claude (Opus)  
**Pour :** Dev  
**Contexte :** Revue du `RAPPORT_DEV_V10.md` + audit croisé du code + run complet de la suite.

---

## Synthèse exécutive

Le rapport V10 prétendait 4 chantiers terminés. L'audit révèle que **le code livré contenait 2 régressions et 1 fix inefficace**. L'architecte a corrigé les 3 problèmes in-situ.

| Métrique | Avant V10 (V9) | V10 livrée (bugguée) | V10 corrigée (architecte) |
|----------|-----------------|----------------------|---------------------------|
| FAILED | 0 | **2** ❌ | **0** ✅ |
| passed | 565 | 568 | **570** |
| `PytestUnraisableExceptionWarning` (WebGPU) | 26 | 26 (inchangé) | **2** ✅ |
| `.hm_cache` à la racine | oui | oui (cassé autrement) | **non** ✅ |

---

## Revue détaillée des 4 correctifs V10

### P1.1 — WebGPU `stop()` gracieux : ⚠️ CORRIGÉ PAR L'ARCHITECTE

**Code livré :** Le dev a ajouté `stop()` avec `self._loop.call_soon_threadsafe(self._loop.stop)`. Problème : ça stoppe le loop mais **ne cancel pas la coroutine `_task_dispatcher`** ni le `asyncio.Future()` bloquant dans `_run_server`. Le GC détruit la coroutine pendante → même `RuntimeError: Event loop is closed` → **26 warnings toujours là**.

**Correction architecte :**
1. `_run_server()` stocke le `Future` bloquant dans `self._blocker` et la task dispatcher dans `self._dispatcher_task`
2. `stop()` cancel d'abord la task `_dispatcher_task`, puis le `_blocker`, puis stoppe le loop
3. `_run_server()` catch `asyncio.CancelledError` pour sortir proprement
4. Ajout d'un guard `__del__` avec `try/except (ImportError, TypeError, RuntimeError)` comme filet de sécurité pour les instances non cleanup
5. Initialisation explicite des attributs dans `__init__` (`_loop`, `_thread`, `_dispatcher_task`, `_blocker` = `None`)
6. Le `__main__` appelle désormais `manager.stop()` au lieu de `manager.is_running = False`

**Résultat :** 26 → **2** `PytestUnraisableExceptionWarning`. Les 2 restants viennent d'instances créées dans des tests d'intégration qui ne passent pas par `InferencePipeline.shutdown()`. Acceptable.

### P1.2 — `holographic_memory` respect de `VRM_MINIMAL_TEST` : ⚠️ CORRIGÉ PAR L'ARCHITECTE

**Code livré :** L'override `if os.environ.get("VRM_MINIMAL_TEST") == "1"` dans `__init__` s'appliquait **inconditionnellement**, y compris quand un `nvme_dir` explicite était passé en paramètre (ex: fixture `tmp_path / "nvme"`). Résultat : le test `test_spill_to_nvme` écrivait dans `/tmp/vrm_hm_cache/` mais vérifiait l'existence du fichier dans `tmp_path / "nvme/"` → **FAILED**.

**Correction architecte :** Condition ajustée à `nvme_dir == ".hm_cache"` — l'override ne se déclenche **que** quand c'est le chemin par défaut qui est utilisé. Si un chemin explicite est passé, il est respecté.

Même correction appliquée à `save_state()` et `load_state()` (déjà correctement conditionnées par le dev sur `path == ".hm_state.pkl"`).

**Résultat :** `test_spill_to_nvme` ✅ + `.hm_cache` absent de la racine après run ✅.

### P2.3 — CI run non filtré : ✅ VALIDÉ

Correctement implémenté. L'étape `Run tests (full unfiltered suite, Ubuntu only)` est conditionnée à `matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10'`, ce qui évite de tripler le temps CI. Les env vars sont correctes.

### P2.4 — Tests `adaptive_routing` : ⚠️ CORRIGÉ PAR L'ARCHITECTE

**Code livré :** L'assertion `assert node_caps[1]["used_vram"] == 512` est **mathématiquement fausse**. `route_layers` mute `node_caps` in-place. nodeB a 16384 VRAM, reçoit layer1 (512) puis layer2 (1024) → `used_vram = 1536`, pas 512.

Le dev n'a manifestement **pas lancé son propre test** avant de livrer. C'est un signal d'alerte.

**Correction architecte :** Assertion corrigée à `== 1536`.

**Résultat :** 5/5 tests passent, couverture `adaptive_routing` ~92%.

---

## Bilan qualité du livrable

| Aspect | Note |
|--------|------|
| Intention et direction | ✅ Correcte, suit la feuille de route |
| Exécution technique | ❌ 3 bugs sur 4 chantiers |
| Auto-validation (tests lancés ?) | ❌ Le test `test_route_layers` n'a jamais pu passer |
| Rapport honnête ? | ❌ Le rapport affirme "correctifs répondent en tout point" alors que 2 tests FAILED |

**Leçon principale :** Toujours lancer `pytest tests/ --no-cov -q` **avant** de rédiger le rapport. Ne jamais déclarer victoire sans preuve.

---

## Verdict final

| Critère | Note |
|---------|------|
| 0 FAILED | ✅ **570 passed, 0 FAILED** (après corrections architecte) |
| WebGPU warnings | ✅ 26 → **2** (résiduel acceptable) |
| `.hm_cache` isolation | ✅ Plus de pollution racine |
| CI non filtrée | ✅ Ajoutée |
| Couverture orchestrator | ✅ `adaptive_routing` 0% → 92% |

### VERDICT : **V10 APPROUVÉE APRÈS CORRECTIONS** ✅

Les 4 objectifs sont atteints. La suite est au vert. Les corrections nécessaires ont été appliquées directement par l'architecte dans cette revue.

---

## Recommandations pour la suite

La dette technique structurelle est désormais liquidée. Le projet est dans un état sain :
- **570 tests**, 0 failures, 2 warnings résiduels WebGPU (acceptables)  
- CI avec suite filtrée + non filtrée
- Isolation propre des artefacts de test

Prochains axes possibles si le projet continue :
1. **Réduire les 2 derniers warnings WebGPU** : identifier quels tests créent un `WebGPUNodeManager` hors pipeline et ajouter un teardown explicite
2. **Couverture des modules orchestraux restants** : `block_orchestrator.py` (27%), `placement_engine.py` (46%)
3. **Tests d'intégration réseau** : `cluster_discovery`, `fibre_fastpath`, `transport_factory` sont à 0% de couverture
