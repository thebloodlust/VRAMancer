# FEUILLE DE ROUTE ARCHITECTE -> DEV : Production-Ready Sprint

## Contexte
L'architecte a validé le hotfix `get_pipeline()` et donné le feu vert sur les Sprints A & B.
Le projet est à ~70% fonctionnel. Il reste 4 chantiers pour atteindre le statut "production-ready".

---

## TÂCHE PRIORITAIRE : `scripts/validate_platform.py`

Créer un script unique cross-platform qui valide que VRAMancer fonctionne sur n'importe quelle machine.
**Objectif : une seule commande = un verdict clair pass/fail.**

### Spécification détaillée

```
python3 scripts/validate_platform.py
```

Le script doit enchaîner ces 7 étapes et produire un rapport JSON :

| Étape | Action | Critère pass |
|-------|--------|-------------|
| 1. Env Check | Détecter OS, Python version, GPU (cuda/rocm/mps/cpu), RAM totale | Affichage de la matrice |
| 2. Import Check | Importer tous les modules `core/` un par un | 0 ImportError |
| 3. Unit Tests | Lancer `pytest tests/ -m smoke -q` | 100% pass |
| 4. API Boot | Démarrer `production_api` en subprocess, attendre `/health` | HTTP 200 en < 10s |
| 5. Inference Smoke | POST `/api/generate` avec prompt court (mode stub OK) | Réponse non-vide |
| 6. Concurrency Light | 10 requêtes concurrentes sur `/api/generate` | 0 crash, 0 timeout 30s |
| 7. Rapport | Écrire `validation_report.json` | Fichier généré |

### Contraintes techniques
- **Zéro dépendance externe** hors ce qui est déjà dans requirements.txt
- Le script doit fonctionner avec `VRM_MINIMAL_TEST=1` (mode stub) ET sans (mode réel)
- Utiliser `subprocess` pour lancer l'API, `urllib.request` pour les requêtes HTTP (pas de `requests` obligatoire)
- Tuer proprement le subprocess API à la fin (même en cas d'erreur)
- Chaque étape doit avoir un try/except avec logging explicite, pas de crash silencieux
- Le rapport JSON doit contenir : timestamp, OS, Python version, GPU détecté, résultat par étape (pass/fail/skip + détail erreur)

### Wrappers par OS
- `validate.sh` (macOS/Linux) : `#!/bin/bash` -> `python3 scripts/validate_platform.py "$@"`
- `validate.bat` (Windows) : `python scripts\validate_platform.py %*`

---

## CHANTIER 2 : Faire passer les tests existants

Actuellement les tests core (`test_pipeline.py`, `test_api_production.py`) passent.
Le marker `@pytest.mark.chaos` a été ajouté aux tests lourds pour ne pas bloquer la CI.

**Action requise :**
- S'assurer que `pytest tests/ -m "not slow and not chaos" -q` passe à 100% en mode `VRM_MINIMAL_TEST=1`
- Vérifier qu'aucun fichier de test n'utilise `InferencePipeline.load(...)` directement (c'est le bug singleton qu'on a corrigé : toujours utiliser `get_pipeline().load(...)` ou `InferencePipeline().load(...)`)

---

## CHANTIER 3 : Combler les 226 exceptions silencieuses (progressif)

L'architecte avait identifié ~226 blocs `except Exception: pass` dans le code.
**Pas besoin de tout faire d'un coup.** Priorité aux fichiers hot-path :

1. `core/inference_pipeline.py`
2. `core/backends.py`
3. `core/transfer_manager.py`
4. `core/scheduler.py`
5. `core/stream_manager.py`

Pour chaque `except Exception: pass` dans ces fichiers :
- Remplacer par `except Exception: logger.debug("...", exc_info=True)` au minimum
- Si l'exception est critique (perte de données, corruption mémoire), utiliser `logger.error`

---

## CHANTIER 4 : Commiter proprement et pousser

Une fois validate_platform.py créé et les tests validés :
```bash
git add scripts/validate_platform.py validate.sh validate.bat
git add benchmarks/ tests/test_chaos_concurrency.py
git commit -m "feat(V5): production-ready validation suite + benchmark + chaos tests"
```

---

## ÉTAT DU PROJET (pour référence du Dev)

| Composant | État | Action restante |
|-----------|------|-----------------|
| API Flask | Fonctionnel | Aucune |
| Pipeline inférence | Fonctionnel (stub + HF) | Benchmark réel Sprint A |
| Split multi-GPU | Implémenté | Tester sur vrai hardware |
| Transport P2P | CPU-staged OK | Rust natif Sprint C |
| VRAM Lending | Architecture prête | Câbler sur cluster réel |
| Dashboard Web | Fonctionnel | Aucune |
| Monitoring Prometheus | Fonctionnel | Aucune |
| Sécurité HMAC/JWT | Fonctionnelle | Aucune |
| Tests | ~8000 lignes, ~14% couverture | Monter à 50%+ |
| validate_platform.py | **À CRÉER** | **PRIORITÉ 1** |

---

*Instructions de l'Architecte Claude, transmises au Dev pour exécution immédiate.*
*Date : 19 mars 2026*
