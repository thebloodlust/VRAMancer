# Audit — core/backends_webgpu.py

## Résumé
Backend WebGPU distribué expérimental pour inférence mobile/navigateur utilisant des nœuds de calcul attachés par WebSocket.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~380+ |
| **Qualité** | 🔴 Expérimental |
| **Sécurité** | 🔴 Critique |
| **Performance** | 🔴 Critique |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 CRITIQUE | **Pas d'authentification client** : WebSocket accepte sans validation |
| 🔴 CRITIQUE | **Perte de scale de quantization** : INT8 scale non utilisé en déquantification |
| 🔴 CRITIQUE | **Memory leak** : event loop créé mais jamais arrêté |
| 🔴 HAUTE | Nombres magiques hardcodés (layer ID modulo 12, retry 3, timeouts) |
| 🟡 MOYENNE | Reconstruction de tenseur fragile (assume correspondance exacte de forme) |
| 🟡 MOYENNE | Stubs renvoyant des strings au lieu de tenseurs en cas d'erreur |

## Verdict
⚠️ **NE PAS UTILISER EN PRODUCTION** — marqué "Production Mode" mais clairement expérimental
