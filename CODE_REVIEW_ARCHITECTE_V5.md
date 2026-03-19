# CODE REVIEW ARCHITECTE (Claude) — Sprint A & B

## Verdict global : Direction validée, mais le code nécessite 7 corrections avant le feu vert.

Le Dev a bien capté le message de l'Analyse V5 : on passe à la preuve par la métrique. La structure des deux fichiers est saine et la direction correcte. **Cependant**, en l'état, ces scripts ne produiront pas de résultats fiables en production. Voici les points à corriger :

---

### SPRINT A — `benchmarks/run_bench.py` : 4 problèmes

| # | Sévérité | Problème | Correction attendue |
|---|----------|----------|---------------------|
| **A-1** | **BLOQUANT** | `os.environ["VRM_MINIMAL_TEST"] = "0"` est posé **après** les imports de `core/`. Or les modules `core/` lisent cette variable **au moment de l'import** (top-level). Le stub est déjà activé quand on arrive à la ligne 31. | Déplacer le `os.environ` **avant** tous les imports `core/`, ou mieux : le passer via CLI (`VRM_MINIMAL_TEST=0 python benchmarks/run_bench.py`). |
| **A-2** | **MAJEUR** | `peak_vram` est mesuré **une seule fois** après la boucle (`monitor.vram_usage()`). Ce n'est pas le pic, c'est la valeur finale. Le vrai pic peut survenir pendant l'inférence. | Lancer un thread de sampling VRAM en background (polling toutes les 100ms) et capturer le `max()`. |
| **A-3** | **MAJEUR** | Aucune métrique de **percentile** (P50, P95, P99). La moyenne seule est trompeuse — une seule requête à 30s tire la moyenne vers le haut masquant que P50 est à 0.5s. | Calculer les percentiles via `sorted()` ou `statistics.quantiles()`. |
| **A-4** | **MINEUR** | `select_backend` et `resolve_config` sont importés mais jamais utilisés — imports morts. | Supprimer les imports inutiles. |

### SPRINT B — `tests/test_chaos_concurrency.py` : 3 problèmes

| # | Sévérité | Problème | Correction attendue |
|---|----------|----------|---------------------|
| **B-1** | **BLOQUANT** | `exceptions.append(e)` depuis 50 threads concurrents sur une **list Python non thread-safe**. En cas de vraie race condition, c'est le collecteur lui-même qui corrompt la liste. | Utiliser une `queue.Queue()` ou un `threading.Lock` autour de l'append. |
| **B-2** | **MAJEUR** | `test_pipeline_oom_simulation` — le `pytest.raises(Exception)` avec un `try/except/raise` interne est un **anti-pattern**. Si `generate()` ne raise pas, le test **passe silencieusement** au lieu d'échouer (pas d'assertion sur l'absence d'exception). | Enlever le `try/except` interne. Laisser `pytest.raises` capturer directement. Si le pipeline ne raise pas d'OOM, le test doit explicitement `pytest.fail()`. |
| **B-3** | **MINEUR** | Pas de marker `@pytest.mark.slow` ni `@pytest.mark.chaos`. Les 50 threads seront lancés à chaque `pytest` standard et ralentiront la CI. | Ajouter `@pytest.mark.slow` conformément aux conventions `pytest.ini`. |

---

### Décision architecte

**Feu vert conditionnel :** la direction est la bonne, les deux fichiers sont bien structurés. Mais les 3 points **BLOQUANTS** (A-1, B-1, B-2) doivent être corrigés avant de merger. Les 4 points MAJEUR/MINEUR sont à traiter dans la même PR si possible.

**Ne pas lancer les runs intensifs tant que A-1 n'est pas corrigé** — sinon on benchmarke le stub et pas le vrai pipeline.

Pour la question du Dev sur "Sprint C (Rust) tout de suite ?" : **Non.** D'abord on a des métriques Python fiables (Sprint A corrigé), ensuite on identifie les goulots d'étranglement réels via le profiler (`core/layer_profiler.py`), et **seulement après** on porte en Rust les fonctions qui le justifient par la donnée. Pas d'optimisation prématurée.

*— L'Architecte, en attente du patch.*
