# Audit — core/api/ (6 modules)

## api/__init__.py (~15 LOC) — ✅ Bon
Re-exports propres avec `__all__`.

## api/batch_inference.py (~300 LOC) — ✅ Bon
Batching de requêtes concurrent avec fenêtre temporelle. Thread-safe.
- ⚠️ `time.sleep()` ajoute latence aux requêtes simples
- ⚠️ Pas de priorisation des requêtes

## api/circuit_breaker.py (~200 LOC) — ✅ Excellent
Pattern circuit-breaker propre. Machine à états, thread-safe, context manager.
- Aucun problème détecté.

## api/registry.py (~200 LOC) — ⚠️ Mixte
Registre singleton thread-safe pour pipeline d'inférence.
- 🟡 ClusterDiscovery démarré dans `__init__` même si non nécessaire
- 🟡 `generate_stream()` fallback mot-par-mot naïf (split sur espaces)
- 🟡 Pas de vérification auth avant `generate()`/`infer()`

## api/routes_ops.py (~350 LOC) — ⚠️ Mixte
Blueprint Flask pour health/readiness/GPU/system.
- 🟡 Pas d'authentification sur les routes ops
- 🟡 Version fallback hardcodée '0.2.4' vs `core.__version__`
- 🟡 `psutil.cpu_percent(interval=0.1)` bloque 100ms par requête
- 🟡 `/api/nodes` POST accepte enregistrement HTTP non validé

## api/validation.py (~70 LOC) — ✅ Bon
Validation d'entrée pour endpoints API. Bounds checking et coercion de type.
- ⚠️ Limites hardcodées (max_tokens 1-4096)
- ⚠️ Token counting par whitespace : inexact pour CJK
