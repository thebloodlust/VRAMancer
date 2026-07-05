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
`benchmarks/bench_context_scaling.py`, **profil coding** (`VRM_N_CTX=16384`). TTFT (proxy
1 token) + débit décode (~63 tok).
| Prompt (~tok) | TTFT | décode tok/s |
|---|---|---|
| 4 000 | 0.77 s | ~230 |
| 8 000 | 1.50 s | ~234 |
| 12 000 | 2.27 s | ~226 |
| 16 000 | 3.05 s | ~254 |
| 32 000 | — | **rejeté proprement** (400 `context_length_exceeded`, message actionnable) |

**Constat C4** : la cause du plafond 4096 était un **`n_ctx=4096` hardcodé** dans
`backends_llamacpp.py` (pas la découpe en slots — hypothèse réfutée par la mesure « 8K skip »).
Corrigé : **`VRM_N_CTX`** (profil coding = 16384). TTFT ~linéaire (0.77→3.05 s de 4K→16K),
décode stable ~230-254 tok/s (MoE A3B). L'overflow (>16384) renvoie une **erreur 400 claire**,
pas de crash (guardrail C4.3 validé en direct). Pour plus de contexte : `VRM_N_CTX=32768`
si la VRAM le permet.

## Palier A1 (rappel)
❌ Path 2 (forward manuel) NE PASSE PAS : sortie dégénérée, bug `cache_position` (prouvé
single-GPU). accelerate (Path 1) = voie de prod. Détails : `RESULTAT_PALIER_A1.md`.
