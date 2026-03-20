# Audit — core/metrics.py

## Résumé
Registre Prometheus et serveur HTTP non-bloquant. Définit 50+ counters/gauges/histograms pour inférence, mémoire, scheduling, lending pool.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~150 |
| **Qualité** | ✅ Bon |
| **Sécurité** | 🔴 Critique |
| **Performance** | 🟡 Risque de cardinality bomb |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 CRITIQUE | **Port 9108 exposé par défaut** sans authentification — fuite d'informations + DoS |
| 🟡 MOYENNE | Cardinalité illimitée des labels — explosion mémoire possible |
| 🟡 MOYENNE | `counter_value()` dépend de structure interne prometheus_client |
| 🟡 MOYENNE | `WEBGPU_CONNECTED_CLIENTS` défini conditionnellement mais dans `__all__` |
| 🟢 BASSE | Pas de mécanisme de shutdown du serveur métriques |

## Recommandation
Ajouter `VRM_METRICS_BIND_ADDRESS=127.0.0.1` pour limiter au localhost.
