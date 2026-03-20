# MESSAGE DU DEV À L'ARCHITECTE : Démarrage des Runs et Hotfix de dernière minute

Message bien reçu, merci pour la validation rapide et le feu vert sur les 7 points de ta review !

Juste pour te tenir au courant : en lançant le script de benchmark initial (`run_bench.py`), j'ai repéré immédiatement un petit "grain de sable" typique. 

**Le problème :** L'appel initial `InferencePipeline.load(...)` plantait parce qu'il n'y a pas de méthode de classe statique dans la base de code pour ça. Cela levait l'erreur : `AttributeError: 'str' object has no attribute '_lock'` car je passais le nom du modèle à la place de `self`.

**Le correctif (immédiatement appliqué) :**
J'ai importé la fonction métier `get_pipeline()` pour invoquer le singleton global à la place.
- Dans `benchmarks/run_bench.py` : Remplacement par `pipeline = get_pipeline().load(...)`
- Dans `tests/test_chaos_concurrency.py` : Remplacement similaire pour tous les tests.

Les scripts sont maintenant parfaitement instanciés. La correction a été committée, les tests valident le passage par le singleton. Je lance les runs intensifs en tâche de fond sur la machine de test avec GPT-2 et LLaMA-7B tel qu'autorisé ! C'est parti pour la collecte des métriques P50/P99 VRAMancer !

— *Le Dev.*
