# Audit — core/hierarchical_memory.py

## Résumé
Gestion mémoire hiérarchique 7 niveaux (L1→L7) implémentant un hyperviseur VRAM distribué. Supporte GPU VRAM (L1), P2P inter-GPU (L2), GPU distants RDMA (L3), RAM hôte épinglée (L4), NVMe (L5), RAM distante (L6), et fallback réseau/disque (L7).

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~800 |
| **Qualité** | ⚠️ Mixte |
| **Sécurité** | 🔴 Critique |
| **Performance** | 🟡 Contention de lock |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 CRITIQUE | **Vulnérabilité pickle** : `pickle.load()` dans `load_from_nvme()` — RCE si cache modifié |
| 🔴 HAUTE | Fuite de threads autosave (daemon threads sans shutdown) |
| 🟡 MOYENNE | Lock contention : toutes migrations tiennent `_lock` pendant le déplacement physique |
| 🟡 MOYENNE | Bridge CXL incomplet (stubs) |
| 🟡 MOYENNE | Pas de `__del__` ou context manager pour arrêter les threads |

## Recommandation
Remplacer pickle par protobuf/msgpack ; ajouter vérification ACL sur fichiers NVMe.
