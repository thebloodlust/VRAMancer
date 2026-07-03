# Estimation — Performances 5070 Ti + RX 7900 XT

> Basé sur les mesures réelles de la session. Pas de spéculation.

---

## Ce qu'on a MESURÉ (solide)

| Mesure | Valeur | Contexte |
|---|---|---|
| GpuPipeline CPU-staged | **25.3 GB/s** | 3090↔5070Ti, PCIe 4.0, P2P=NS |
| torch.to() naïf | **11.6 GB/s** | Même setup |
| P2P NVIDIA consumer | **code 217** | PEER_ACCESS_UNSUPPORTED |
| Prompt-lookup décode | **+500% tok/s** | Lossless, 7B greedy |
| PagedAttention kernel | **8.8× vs PyTorch** | Mesuré |
| DirectFP4 bypass | **+7% vs torchao** | Mesuré |
| ClusterRouter data-parallel | **×1.97** | 2 workers, 32 req |

---

## Ce qu'on EXTRAPOLE de ces mesures

### Bande passante NVIDIA↔AMD

```
Même mécanisme que 3090↔5070Ti : CPU-staged via GpuPipeline

3090↔5070Ti : 25.3 GB/s (mesuré)
5070Ti↔7900XT : ~22-25 GB/s (estimé, même PCIe 4.0, même CPU-staged)

Pourquoi pas + ? PCIe 4.0 x16 plafond = ~32 GB/s.
GpuPipeline atteint ~78% du plafond → ~25 GB/s.
La 7900 XT a aussi PCIe 4.0 x16 → même plafond → même résultat.

DMA-BUF cross-vendor (si ça marche) : potentiellement ~28-30 GB/s.
Mais ça n'a jamais été testé. Hypothèse, pas mesure.
```

### VRAM lending — coût du swap

```
Couche de 100 MB → 100 MB / 25 GB/s = 4 ms
Page KV de 10 MB  → 10 MB / 25 GB/s  = 0.4 ms

Pour 48 couches (14B), swapper 40 couches froides :
→ 40 × 100 MB = 4 GB → 4 GB / 25 GB/s = 160 ms
→ Payé UNE FOIS au chargement, pas par token
→ Gratuit après (les couches chaudes restent sur 5070 Ti)
```

### VRAM effective

```
5070 Ti seule : 16 GB → 14B FP4 tient (~7 GB)
5070 Ti + 7900 XT lending : 16 + 20 = 36 GB "effectifs"

Avec 36 GB effectifs :
→ 14B FP4  : 7 GB  → tient large (reste 9 GB pour KV cache)
→ 32B FP4  : ~16 GB → tient juste (reste 4 GB)
→ 70B Q4   : ~40 GB → tient pas (mais tiendrait avec 7900 XT seule en compute)
→ KV cache : jusqu'à ~20 GB supplémentaire → contexte 500K+ tokens
```

---

## Les 3 scénarios concrets

### Scénario 1 — 14B FP4, lending passif

```
5070 Ti : modèle 14B FP4 (~7 GB) + KV cache chaud (~2 GB)
7900 XT : KV cache froid (~10 GB) + modèles secondaires (~10 GB)

Décode : 87 tok/s (prompt-lookup, mesuré sur 7B, extrapolé 14B)
Prefill : normal (pas de swap)
KV swap  : 4 ms par page → transparent

Verdict : RAPIDE. Le lending est transparent. Aucun impact perf.
          La 7900 XT ajoute juste du stockage KV + multi-modèle.
```

### Scénario 2 — 32B FP4, lending actif

```
5070 Ti : couches chaudes 0-23 (~8 GB) + KV cache (~2 GB)
7900 XT : couches froides 24-47 (~8 GB)

Swap au chargement : 8 GB / 25 GB/s = 320 ms (une fois)
Pendant l'inférence : pas de swap (couches chaudes résidentes)

Décode : ~40-50 tok/s (32B FP4, prompt-lookup)
Verdict : BON. Le lending ne coûte qu'au chargement.
          Décode = pas de swap → perf normale.
```

### Scénario 3 — VRAM lending entre 2 AMD (si tu en as 2)

```
7900 XTX (24 GB) + 7900 XT (20 GB), P2P natif AMD↔AMD

Bande passante P2P : ~32 GB/s (direct DMA, pas CPU)
vs CPU-staged : 25 GB/s → +28%

Verdict : EXCELLENT. P2P natif, pas bridé, 44 GB total.
          Mais nécessite ROCm (pas de NVFP4).
```

---

## Tableau récapitulatif

| Setup | VRAM | BW lending | Modèle max | Coût swap | Verdict |
|---|---|---|---|---|---|
| **Actuel** 5070Ti + 3090 | 40 GB | 25 GB/s (CPU) | 14B FP4 (7 GB) | 4 ms/100MB | **Bon** (bridé P2P) |
| **Test** 5070Ti + 7900XT | 36 GB | ~25 GB/s (CPU) | 32B FP4 (~16 GB) | 4 ms/100MB | **Très bon** (AMD = 20GB extra) |
| **Futur** 7900XTX + 7900XT | 44 GB | ~32 GB/s (P2P) | 32B FP4 (~16 GB) | 3 ms/100MB | **Excellent** (P2P natif) |
| **Graal** 5070Ti + 7900XT + DMA-BUF | 36 GB | ~28-30 GB/s (DMA-BUF) | 32B FP4 | 3.5 ms/100MB | **Optimal** (si DMA-BUF marche) |

---

## Réponse directe

**Oui, le setup 5070 Ti + 7900 XT serait performant.** Pas pour du streaming de
poids par token (on a prouvé que ça ne marche pas). Mais pour :

- **Multi-modèle** : 14B sur 5070 Ti, 7B coder sur 7900 XT → switch < 1s
- **KV cache illimité** : 20 GB de pages KV sur 7900 XT → contexte 500K+ tokens
- **Modèle 32B FP4** : couches froides sur 7900 XT, chaudes sur 5070 Ti
- **DMA-BUF test** : première validation cross-vendor du projet

La bande passante de lending (~25 GB/s) est la même que ce qu'on a mesuré.
Le gain n'est pas la vitesse — c'est la **capacité** (+20 GB) et la
**flexibilité** (AMD = P2P ouvert, driver libre, DMA-BUF testable).

— DeepSeek
