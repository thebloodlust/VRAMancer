# Audit — core/backends.py

## Résumé
Couche d'abstraction unifiée pour les backends LLM. Implémente le pattern factory (`select_backend()`), support KV-cache, backend HuggingFace, et classe de base abstraite pour tous les backends.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~1000+ |
| **Qualité** | ⚠️ Mixte |
| **Sécurité** | 🔴 Risques critiques |
| **Performance** | 🔴 Problèmes majeurs |

## Classes/Fonctions clés
- `BaseLLMBackend` (ABC)
- `HuggingFaceBackend`
- `KVCacheBlock` (wrapper nn.Module pour inférence multi-GPU avec KV-cache)
- `select_backend(model_name, backend, num_gpus)` — factory
- `_extract_model_components()` — extraction embedding/norm/lm_head

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 CRITIQUE | **vLLM forcé sans fallback** : `select_backend("auto")` crashe si vLLM absent |
| 🔴 CRITIQUE | **Thread-safety violation** : `asyncio.run()` dans contexte sync (WebGPU intercept) |
| 🔴 CRITIQUE | **Memory leak** : event loop asyncio créé par inférence, jamais détruit |
| 🔴 HAUTE | **trust_remote_code=True** activé inconditionnellement dans vLLM |
| 🟡 MOYENNE | `print('EXCEPTION IN INFER WITH KV_CACHE:')` dans code production |
| 🟡 MOYENNE | Résolution de device dupliquée 4+ fois par forward pass |
| 🟡 MOYENNE | Sérialisation CPU→GPU→CPU coûteuse (numpy.tobytes roundtrip) |
| 🟡 MOYENNE | Pas de validation d'entrée pour prompts/noms de modèles |

## Recommandations
1. Ajouter `try/except` avec fallback HuggingFace pour `select_backend("auto")`
2. Remplacer `asyncio.run()` par `asyncio.run_coroutine_threadsafe()`
3. Supprimer les `print()` de debug
4. Cacher la résolution de device au moment du chargement
