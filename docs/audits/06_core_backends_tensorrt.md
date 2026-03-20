# Audit — core/backends_tensorrt.py

## Résumé
Intégration moteur d'inférence TensorRT pour exécution GPU compilée avec transferts VRAM zero-copy.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~150 |
| **Qualité** | ⚠️ Acceptable |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | ✅ Bon (zero-copy) |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 HAUTE | **Assumption forme de sortie** : `output_shape = inputs.shape` incorrect pour de nombreux modèles |
| 🟡 MOYENNE | Bindings assumés dans l'ordre du moteur |
| 🟡 MOYENNE | Chemin ONNX non validé |
| 🟡 MOYENNE | Module jamais importé — code mort |
| 🟢 BASSE | workspace_mb hardcodé à 2048 MB |
