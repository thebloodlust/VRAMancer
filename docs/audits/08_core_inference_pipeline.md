# Audit — core/inference_pipeline.py

## Résumé
Orchestrateur principal connectant tous les sous-systèmes VRAMancer : backend → scheduler → transfer manager → GPU monitoring → metrics.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~1000+ |
| **Qualité** | ⚠️ Mixte |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Overhead WebGPU |

## Classes/Fonctions clés
- `InferencePipeline` : `load()`, `generate()`, `infer()`, `_protected_generate()`, `_protected_infer()`
- Singleton global : `get_pipeline()`, `reset_pipeline()`

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 HAUTE | **WebGPU toujours démarré** : `WebGPUNodeManager(port=8560)` lancé sans flag |
| 🟡 MOYENNE | Port 8560 hardcodé (conflit si instances multiples) |
| 🟡 MOYENNE | Thread de rééquilibrage jamais arrêté (fuite) |
| 🟡 MOYENNE | `select_backend()` force vLLM → crash si pas installé |
| 🟡 MOYENNE | Pas de rate limiting sur `generate()` concurrent |

## Couverture de test
✅ Bonne : `test_pipeline.py`, `test_e2e_pipeline.py`, `test_fault_pipeline.py`
