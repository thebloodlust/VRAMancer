# Audit — core/transfer_manager.py

## Résumé
Transfert GPU-to-GPU haute performance avec P2P, NCCL, CPU-staged et support cross-vendor. Détection de topologie et sélection transport optimal.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~800+ |
| **Qualité** | ⚠️ Mixte |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Overhead de synchronisation |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 HAUTE | **Code Rust incomplet** : lignes 649-674 commentées, bypass WIP |
| 🔴 HAUTE | `vramancer_rust.direct_vram_load()` non défini dans l'extension |
| 🟡 MOYENNE | Commentaires en français ("Bypass Nvidia", "dégradé") |
| 🟡 MOYENNE | NCCL sans chiffrement (si mode cluster) |
| 🟡 MOYENNE | Stats `_total_bytes` / `_total_time_s` peuvent overflow |
| 🟡 MOYENNE | Probing topologie O(n²) pour tous les pairs GPU |
| 🟢 BASSE | KV cache assume forme avec `if k_cache.dim() > 1` fragile |
