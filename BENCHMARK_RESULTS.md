# BENCHMARK_RESULTS — source unique des chiffres de performance

> Référence unique (architecte §2, post-A1). Tout chiffre de perf ailleurs (README,
> docs) doit correspondre à ce fichier. Pas de spéculation, que du mesuré.
> Matériel : RTX 3090 (24 GB, Ampere) + RTX 5070 Ti (16 GB, Blackwell), Proxmox VM,
> VFIO passthrough, pas de NVLink, P2P direct **indisponible** (`cuCtxEnablePeerAccess`
> → 217 `PEER_ACCESS_UNSUPPORTED`).

## Le chiffre de prod bf16 multi-GPU : **5.41 tok/s**
`Qwen2.5-14B-Instruct`, 2 GPU, **accelerate** `device_map="auto"`, greedy.
- Remplace **partout** les valeurs contradictoires antérieures « 6.0 » et « 16.1 » (obsolètes).
- Méthodo exacte : `device_map="auto"`, `max_memory` par GPU, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `VRM_DISABLE_TURBO=1`.
- Source : `benchmarks/test_a1_accelerate_baseline.py` (palier A1, cf. `RESULTAT_PALIER_A1.md`).
- C'est le chemin de PROD (A1 a acté accelerate ; le forward manuel Path 2 est déprécié).

### Fix OOM au chargement (réutilisable, non trivial)
Sans `max_memory`, le 14B OOM au chargement : `_initialize_missing_keys` appelle
`init.normal_(weight.float())` → **upcast fp32** temporaire sur un GPU déjà plein → OOM.
Correctif : `max_memory` par GPU (laisse de la marge) + `expandable_segments:True`. À
conserver dans tout chemin de chargement multi-GPU tight-fit.

## Autres chiffres mesurés
| Config | Modèle | tok/s | Notes |
|---|---|---|---|
| bf16 2-GPU (accelerate) | Qwen2.5-14B | **5.41** | chemin de prod, méthodo ci-dessus |
| NF4 1-GPU (bitsandbytes) | Qwen2.5-14B | **10.5** | ~70% moins de VRAM ; plus rapide que bf16-2GPU (0 transfert cross-GPU) |
| GGUF Q4_K_M 1-GPU (llama.cpp) | Qwen2.5-7B | **106.8** | — |
| GGUF Q3_K_XL 2-GPU (llama.cpp, MoE) | Qwen3-Coder-Next 80B/3B actif | **~60** | 1er token 66-92 ms ; rapide car ~3B actifs/token (sparsité MoE) |

## Transfert GPU↔GPU (tous CPU-staged — P2P indisponible)
| Méthode | Débit | Notes |
|---|---|---|
| Rust pinned double-buffer (`GpuPipeline`) | ~25 GB/s | CPU-staged, overlappé, gros transferts contigus |
| PyTorch `.to()` | ~11.6 GB/s | CPU-staged naïf (mesuré, 256 MB) |

Pas de chemin DMA P2P sur ce matériel. Un transport plus rapide exigerait NVLink, ou
(cross-nœud) Thunderbolt/USB4 (~16-20 Gbps).

## Cross-nœud / cluster (data-parallel)
| Config | Résultat |
|---|---|
| data-parallel local, threads | ×0.97 (artefact GIL) |
| data-parallel local, **process** (ClusterRouter), 32 req | **×1.97** (work-stealing 16/16) |

## Contexte / agent de code (C4) — Qwen3.6-35B-A3B GGUF Q4, llama.cpp, 2 GPU
`benchmarks/bench_context_scaling.py`. TTFT (proxy 1 token) + débit décode (~63 tok).
| Prompt (~tok) | TTFT | décode tok/s |
|---|---|---|
| 1 000 | 0.31 s | ~305 |
| 2 000 | 0.41 s | ~235 |
| 4 000 | 0.77 s | ~246 |
| ≥ 6 000 | — | rejeté (contexte 4096) |

**Constat C4** : le contexte effectif par requête est **4096 tokens** (continuous batching
= 4 slots × 4096 sur n_ctx 16384) — **trop petit pour un agent de code** (fichiers entiers).
Bon point : l'overflow renvoie une **erreur propre** (`exceed context window of 4096`),
pas de crash (guardrail C4.3 actionnable côté API en amont).

**Reco (à appliquer au prochain restart)** : pour l'usage coding (mono-utilisateur), donner
tout le contexte à une requête — `VRM_CONTINUOUS_BATCHING=0` (1 slot → ~16K/req) et/ou
augmenter `n_ctx` (32768/65536) si la VRAM le permet. Puis re-mesurer 8K/32K/64K.

## Palier A1 (rappel)
❌ Path 2 (forward manuel) NE PASSE PAS : sortie dégénérée, bug `cache_position` (prouvé
single-GPU). accelerate (Path 1) = voie de prod. Détails : `RESULTAT_PALIER_A1.md`.
