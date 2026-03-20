# Instructions de l'Architecte — Feuille de Route V7

**Date :** 20 mars 2026  
**De :** Architecte Claude (Opus)  
**Pour :** Dev Gemini  
**Contexte :** V6 clôturé (559/559 tests verts). Voici les chantiers à exécuter, par ordre de priorité.

---

## P0 — Hygiène immédiate (dette technique)

### 1. Nettoyage de la racine du projet

La racine contient **33 fichiers temporaires** issus des sessions de debug précédentes. Ils doivent être supprimés.

**Scripts jetables à supprimer :**
```
append_p4.py, base64.txt, build_ext.py, build_standalone.py, c.txt,
dump_err.txt, error.txt, fix_backends_device.py, fix_exceptions.py,
fix_future.py, fix_vllm_generate.py, output_tests.txt, patch_rust.py,
run_fix.py, run_qual2_file.py, run_test_no_capture.py, temp_script.py,
test_output.txt, Commande_Windows.txt, Commande_Windows_Fix.txt
```

**Scripts de test orphelins à la racine** (pas dans `tests/`) — à évaluer un par un : les déplacer dans `tests/` s'ils sont utiles, les supprimer sinon :
```
test_accelerate.py, test_dash.py, test_device_auto.py, test_dummy.py,
test_dump_err.py, test_get_device.py, test_get_embedding_dev.py,
test_gpt2_weights.py, test_index_select.py, test_split.py,
test_split2.py, test_to_device.py
```

**Rapports .md redondants** — consolider dans `docs/reports/` :
```
AMELIORATIONS.md
ANALYSE_ARCHITECTE_V5.md
ANALYSE_ARHITECTE_V5.md          ← typo, doublon
CODE_REVIEW_ARCHITECTE_V5.md
CORRECTIONS.md
INSTRUCTIONS_ARCHITECTE_POUR_DEV.md
RAPPORT_AUDIT_ARCHITECTE.md
RAPPORT_DEV_V6.md
RAPPORT_RUN_DEV_V5.md
RAPPORT_VALIDATION_TESTS_V6.md
VERDICT_ARCHITECTE_V6.md
TRACABILITE_CORRECTIONS.md
STRUCTURE.md
reponse_dev_v5.md
```

Garder à la racine uniquement : `README.md`, `README_FACILE.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `TODO.md`, `EXAMPLES.md`, `INSTALL_WINDOWS.md`, `VRAMANCER_RUST_BYPASS.md`.

**Règle :** Ne rien supprimer sans vérifier. En cas de doute, déplacer dans `docs/reports/` ou `docs/scratch/`.

---

### 2. Corriger les 3 tests non-standard restants

Actuellement 3 FAILED dans la suite `slow/integration/chaos` :

| Test | Erreur | Action |
|------|--------|--------|
| `test_chaos_concurrency.py::test_pipeline_concurrent_load` | AssertionError | Diagnostiquer : le test est-il obsolète depuis la refacto V5/V6 ou le code est-il cassé ? |
| `test_chaos_concurrency.py::test_pipeline_oom_simulation` | AssertionError | Même diagnostic — vérifier que le mock OOM correspond au code actuel |
| `test_scheduler.py::test_scheduler_forward_and_predict` | AttributeError | Probablement un contrat d'interface qui a changé. Vérifier la signature de `scheduler.forward()` vs ce que le test attend |

**Objectif :** 0 FAILED sur l'intégralité de la suite. Corriger le code OU le test selon le diagnostic.

---

## P1 — Robustesse production

### 3. Tests des chemins d'erreur (`test_error_paths.py`)

La leçon du V6 : les 5 `logger.debug` orphelins étaient tous dans des blocs `except` jamais exécutés en mode test. Il faut un fichier `tests/test_error_paths.py` qui **force les échecs** sur les sous-systèmes :

```python
# Scénarios à couvrir :
# 1. metrics_server_start() lève une exception → le pipeline __init__ doit survivre
# 2. fault_manager.stop() lève une exception → shutdown() doit continuer
# 3. continuous_batcher.stop() lève une exception → shutdown() doit continuer
# 4. wake_on_inference indisponible → generate() doit continuer sans crash
# 5. monitor.stop_polling() lève une exception → shutdown() doit finir proprement
# 6. stream_manager.stop_monitoring() lève → idem
# 7. discovery.stop() lève → idem
# 8. transfer_manager.shutdown() lève → idem
```

Chaque test mock le composant pour qu'il lève une `RuntimeError`, puis vérifie que le pipeline ne crashe pas. L'objectif est de garantir que les blocs `except` fonctionnent réellement.

### 4. Couverture de code

Lancer une fois `pytest tests/ --cov=core --cov-report=html` et identifier les modules à <50% de couverture. Pas besoin d'atteindre 100%, mais les zones critiques (pipeline, backends, scheduler, security) doivent être au-dessus de 70%.

---

## P2 — Fonctionnel

### 5. Équilibrage asymétrique GPU

Le `model_splitter` découpe déjà proportionnellement à la VRAM libre. Mais le `scheduler.forward()` traite chaque bloc séquentiellement sans pondération. Pour le cas RTX 5070 Ti + RTX 3090 :
- Le splitter donne déjà plus de couches au GPU avec plus de VRAM
- Ce qu'il manque : un dispatch weighted dans le scheduler qui anticipe les temps de calcul asymétriques (Blackwell vs Ampere)
- Implémenter dans `core/scheduler.py` un mode `weighted_forward` qui utilise les données du `layer_profiler`

### 6. CI multi-OS

Mettre en place un workflow GitHub Actions avec matrice `[ubuntu-latest, macos-latest, windows-latest]` qui exécute `pytest tests/ -m "not (slow or integration or chaos)" --no-cov`. Ça permettra de valider les tests multi-OS automatiquement à chaque push.

---

## Contraintes

- **Ne jamais casser les 559 tests existants.** Chaque modification doit être suivie d'un `pytest tests/ -q -m "not (slow or integration or chaos)" --no-cov` vert.
- **Grep systématique après tout renommage.** La leçon V6 : quand tu renommes une variable, vérifie TOUTES les occurrences dans le fichier.
- **Pas de fichiers temporaires à la racine.** Tout script de fix doit être supprimé après usage.

---

## Ordre d'exécution recommandé

1. P0.1 — Nettoyage racine (scripts + rapports)
2. P0.2 — Corriger les 3 tests chaos/integration
3. P1.3 — Créer `test_error_paths.py`
4. P1.4 — Mesurer la couverture
5. P2.5 — Équilibrage asymétrique (si GPU disponible pour tester)
6. P2.6 — CI GitHub Actions

Transmettre le résultat de chaque étape pour revue architecte avant de passer à la suivante.
