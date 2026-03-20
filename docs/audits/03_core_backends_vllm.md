# Audit — core/backends_vllm.py

## Résumé
Intégration backend vLLM pour inférence LLM à haut débit avec parallélisme tensor/pipeline.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~180 |
| **Qualité** | ⚠️ Acceptable |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Polling synchrone |

## Classes/Fonctions clés
- `vLLMBackend(BaseLLMBackend)`
- `load_model()`, `generate()`, `generate_stream()`, `split_model()`

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🟡 MOYENNE | **Memory leak** : Engine jamais déchargé/libéré |
| 🟡 MOYENNE | **Polling synchrone** : busy-wait dans streaming |
| 🟡 MOYENNE | `infer()` lève `NotImplementedError` |
| 🟡 MOYENNE | Paramètres de sampling filtrés silencieusement |
| 🟢 BASSE | Pas de validation de chemin de modèle |

## Couverture de test
⚠️ Minimale — aucun test dédié pour vLLM
