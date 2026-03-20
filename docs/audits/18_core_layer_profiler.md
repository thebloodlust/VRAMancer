# Audit — core/layer_profiler.py

## Résumé
Profiling par couche pour placement GPU optimal via algorithme DP. Mesure latence, mémoire, FLOPS pour chaque couche et GPU.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~750 |
| **Qualité** | ⚠️ Mixte |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🔴 O(n²) complexité DP |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 HAUTE | **compute_optimal_placement() 200+ lignes** — devrait être refactorisé |
| 🔴 HAUTE | **Contraintes VRAM post-hoc** : correction greedy peut violer solution DP |
| 🟡 MOYENNE | Estimation FLOPS approximative (assume structure transformer standard) |
| 🟡 MOYENNE | `expansion = 4` hardcodé pour MLP |
| 🟡 MOYENNE | Profiling lent (10+ itérations × couches × GPUs) |
| 🟢 BASSE | `speed_factor` pourrait être 0 → division par zéro |
