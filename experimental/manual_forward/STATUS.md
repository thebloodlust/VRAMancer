# STATUS — manual_forward (Path 2)

- **État** : DÉPRÉCIÉ (cassé, non réparé). Pas sur le chemin de prod.
- **Ce qui marche** : rien de fiable — sortie dégénérée (bug `cache_position`).
- **Ce qui manque pour repromouvoir** : (1) propager `cache_position` → A1 repasse (sorties
  identiques à accelerate) ; (2) ET un gain mesuré vs accelerate (aujourd'hui : aucun).
- **Code** : `core/backends.py` (`KVCacheBlock`, `_infer_with_kv_cache`,
  `__infer_with_kv_cache_impl`), marqué déprécié + `DeprecationWarning` à l'usage.
- **Chemin de prod à la place** : accelerate `device_map` (bf16), llama.cpp (GGUF).
- **Réf** : `RESULTAT_PALIER_A1.md`, `BENCHMARK_RESULTS.md`, README de ce dossier.
