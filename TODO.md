# VRAMancer Todo

- Wake On Inference
- WebGPU
- Tests Multi OS PC et MAC
- Déterminer/Forcer l'ordre des GPUs (L0 vs L1) : prioriser la RTX 5070 Ti en maître pour profiter de ses cœurs Tensor NVFP4 (architecture Blackwell) et booster les performances de calcul.
- Implémenter un équilibrage de charge asymétrique (Load Imbalance) : répartir le calcul proportionnellement à la puissance (ex: 65% pour la 5070 Ti, 35% pour la 3090) plutôt qu'à 50/50. Objectif : réduire le temps d'attente entre les cartes lors du Prefill, et faire s'envoler les tok/s.
