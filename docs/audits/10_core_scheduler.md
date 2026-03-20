# Audit — core/scheduler.py

## Résumé
Scheduler GPU de production avec allocation VRAM-aware, préchargement prédictif et migration de blocs live.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~500+ |
| **Qualité** | ✅ Bon |
| **Sécurité** | ✅ Aucun risque |
| **Performance** | 🟡 Accounting VRAM estimé |

## Classes/Fonctions clés
- `AllocatedBlock` (dataclass)
- `SimpleScheduler` : `forward()`, `predict()`, `allocate_block()`, `release_block()`, `predict_next_layers()`, `find_alternate_gpu()`, `migrate_block()`

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🟡 MOYENNE | **Accounting VRAM non live** : estimation, pas de tracking réel |
| 🟡 MOYENNE | **Paramètre priority ignoré** dans l'allocation |
| 🟡 MOYENNE | **Préchargement trivial** : `predict_next_layers()` retourne indices séquentiels |
| 🟢 BASSE | Log emoji "🔮 [Swarm Synapse]" non professionnel |
