# Audit — core/compute_engine.py

## Résumé
Moteur d'exécution de couches: exécute des instances nn.Module sur GPU/CPU avec profiling, optimisation torch.compile et gestion d'erreurs.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~400 |
| **Qualité** | ✅ Bon |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Pas d'async |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🟡 MOYENNE | **torch.compile() appelé à chaque inférence** — devrait être caché |
| 🟡 MOYENNE | Profiler overhead élevé (record_shapes=True) |
| 🟡 MOYENNE | Exécution synchrone bloque les threads |
| 🟢 BASSE | Export ONNX : catch Exception trop large |
| 🟢 BASSE | psutil importé inconditionnellement |
