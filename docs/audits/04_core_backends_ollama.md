# Audit — core/backends_ollama.py

## Résumé
Intégration backend Ollama via API REST avec support async et streaming SSE.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~350 |
| **Qualité** | ✅ Bon |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🔴 Fuite de session |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 HAUTE | **Fuite de ressource** : session aiohttp jamais fermée |
| 🟡 MOYENNE | `OLLAMA_HOST` depuis env sans validation d'URL |
| 🟡 MOYENNE | Timeout hardcodé (120s) |
| 🟢 BASSE | `str(inputs)` peut produire une sortie inattendue pour les tenseurs |

## Couverture de test
⚠️ Aucun test dédié
