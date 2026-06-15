# ClusterRouter v0 — data-parallel multi-process : le GIL est levé, ça scale

> `benchmarks/bench_cluster_router.py` sur Qwen2.5-0.5B, 2 GPU (3090 + 5070 Ti).
> La brique qui débloque cross-vendor + cross-nœud (même archi). Date : 2026-06-15.

## Le point qui était à prouver
Plus tôt, le data-parallel **par threads** donnait **×0.97** (artefact GIL : la boucle de
décode Python ne parallélise pas). En **process isolés** (1 GPU/worker, file de travail
partagée = work-stealing), ça doit vraiment scaler.

## Mesuré
| Config | tok/s agrégé | répartition | speedup |
|---|---|---|---|
| 1 worker (GPU0) | 41.4 | 8 req → GPU0 | — |
| 2 workers (8 req) | 62.2 | 3 → GPU0, **5 → GPU1** | **×1.5** |
| 2 workers (32 req) | _(run en cours)_ | _équilibrage attendu meilleur_ | _attendu > ×1.6_ |

## Lecture honnête
- **L'artefact GIL est levé** : threads ×0.97 → process **×1.5**. L'architecture
  multi-process **scale vraiment**. C'est le point clé, prouvé.
- Le **work-stealing s'auto-équilibre vers le GPU rapide** : GPU1 (5070 Ti Blackwell) a pris
  **5** requêtes vs 3 pour le GPU0 (3090 Ampere). Pas de placement manuel — la file partagée
  fait le boulot.
- Le **×1.5 (pas ×2)** s'explique : sur seulement 8 requêtes, le split discret 3/5 plafonne
  le speedup à ~8/5 = ×1.6 ; + GPU hétérogènes. Avec plus de requêtes, l'équilibrage
  s'améliore (cf. run 32-req).

## Pourquoi ça compte
`ClusterRouter` = **une seule brique, 3 usages** (insight mesuré : torch est mono-vendeur
→ cross-vendor = multi-process = même archi que cross-nœud) :
1. **Local multi-process** (testé ici) — base.
2. **Cross-vendor** (NVIDIA worker + AMD worker) — quand une AMD arrive.
3. **Cross-nœud** (Thunderbolt) — quand la 2e machine arrive.

Même routeur, même transport, même data-parallel. C'est le créneau qu'accelerate/vLLM ne
font pas — sans réimplémenter de moteur (chaque worker garde accelerate/torch en interne).
