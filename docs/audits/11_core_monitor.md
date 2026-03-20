# Audit — core/monitor.py

## Résumé
Monitoring GPU de production avec support multi-accélérateur (CUDA, ROCm, MPS), polling continu et export Prometheus. Inclut détection hot-plug GPU.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~850 |
| **Qualité** | ⚠️ Mixte |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Polling bloquant |

## Classes/Fonctions clés
- `GPUMonitor` : `vram_usage()`, `detect_overload()`, `memory_allocated/reserved/total()`, `start_polling()`, `stop_polling()`
- `GPUHotPlugMonitor` — détection hot-plug GPU
- `_rocm_smi_memory()`, `_macos_total_memory()`

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 HAUTE | **Bug `stop_polling()`** : guard `if sys.meta_path is None` non pertinent |
| 🔴 HAUTE | Code après guard inaccessible (dead code) |
| 🟡 MOYENNE | pynvml.nvmlShutdown() pas appelé en cas d'erreur |
| 🟡 MOYENNE | Background polling bloque avec `_lock` pendant `_refresh_all()` |
| 🟢 BASSE | Timeout subprocess hardcodé à 5s |
