# Audit — core/memory_balancer.py, memory_block.py, memory_monitor.py

## memory_balancer.py (~80 LOC) — ✅ Bon
Load balancer GPU basé LRU cache. Alloue les blocs à travers GPUs et rééquilibre.
- ⚠️ `migrate_block()` ne valide pas si src_gpu existe
- ⚠️ `cache_size=2` par défaut très petit
- ⚠️ Pas de validation bounds sur index GPU

## memory_block.py (~30 LOC) — ✅ Minimal
Classe de données pour blocs VRAM avec UUID, taille, affinité GPU et cycle de vie.
- ⚠️ Pas de `__slots__` (gaspillage mémoire)
- ⚠️ Pas de validation `size_mb` (pourrait être négatif)

## memory_monitor.py (~110 LOC) — ✅ Bon
Monitoring RAM système avec détection de pression (low/medium/high/critical).
- ⚠️ Seuils hardcodés (95%, 85%, 70%) — devraient être configurables
- ⚠️ Pas de cache : chaque appel interroge psutil (syscall lent)
