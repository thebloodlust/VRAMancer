# Audit — core/paged_attention.py

## Résumé
Gestionnaire de KV cache par pages inspiré de vLLM PagedAttention. Allocation on-demand, copy-on-write pour beam search, prefix caching, VRAM lending pool overflow.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~900+ |
| **Qualité** | ✅ Excellent |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Contention de lock |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🟡 MOYENNE | **VTP offload incomplet** : intégration avec `hm_manager` assume globals |
| 🟡 MOYENNE | Hash O(n²) dans `try_prefix_cache()` |
| 🟡 MOYENNE | Éviction LRU O(n) — devrait utiliser heap |
| 🟡 MOYENNE | Pool multi-GPU peut échouer silencieusement |
| 🟢 BASSE | `page_size=16` tokens/page hardcodé (défaut vLLM) |
