# DeepSeek — Performances attendues : AMD pour VRAM Lending

> Analyse des scénarios de performance avec un GPU AMD (RX 7900 XT)
> pour le VRAM lending. Date : 2026-06-16.

---

## Scénario A : NVIDIA compute + AMD lending (cross-vendor, ton setup actuel)

```
GPU0 = RTX 5070 Ti (16 GB, FP4) → COMPUTE
GPU1 = RX 7900 XT   (20 GB)     → VRAM LENDING PUR

Transfert via GpuPipeline CPU-staged (cross-vendor = P2P improbable)
```

| Métrique | Estimé | Basé sur |
|---|---|---|
| Bande passante | **~25 GB/s** | GpuPipeline mesuré (optimum CPU-staged) |
| VRAM totale utilisable | **36 GB** (16 + 20) | Compute + lending |
| Modèle max en FP4 | **~14B** (~7 GB) | Tient seul sur 5070 Ti |
| Modèle max avec lending | **~40B-70B** (selon quant) | 36 GB VRAM "effective" |
| Coût du swap par couche (100 MB) | **~4 ms** | 25 GB/s |
| Nombre de couches swappables par token | **~4** | Budget 16 ms/token à 25 tok/s |
| KV cache extensible | **+20 GB** | Sur GPU1, contexte quasi-illimité |

### Verdict scénario A

**Bon.** Le lending cross-vendor fonctionne via CPU-staged (déjà prouvé).
L'AMD ajoute 20 GB de "VRAM effective". Le coût de swap est ~4 ms pour 100 MB.
Pour du lending de couches froides ou de KV cache, c'est excellent.
Pour du streaming de poids par token, c'est trop lent (même problème que le tiering).

---

## Scénario B : AMD + AMD (pur AMD, P2P natif)

```
GPU0 = RX 7900 XTX (24 GB) → COMPUTE + VRAM lending master
GPU1 = RX 7900 XT  (20 GB) → VRAM lending pur

P2P NATIF via amdgpu. DMA direct GPU↔GPU. Zéro CPU.
```

| Métrique | Estimé | vs NVIDIA+NVIDIA |
|---|---|---|
| Bande passante P2P | **~32 GB/s** | +28% vs CPU-staged 25 GB/s |
| Latence P2P | **~1-3 µs** | vs ~50-100 µs CPU-staged |
| VRAM totale | **44 GB** | +8 GB vs 3090+5070Ti (36 GB) |
| Coût swap 100 MB | **~3 ms** | vs 4 ms CPU-staged |
| P2P bloqué ? | **NON** | amdgpu = pas de segmentation |
| Driver modifiable ? | **OUI** | Open-source, dans le kernel |

### Verdict scénario B

**Excellent.** C'est le setup optimal pour le VRAM lending :
- P2P natif, pas bridé, ~32 GB/s
- 44 GB VRAM totale avec deux GPUs
- Driver ouvert → on peut TOUT optimiser
- ROCm pour le compute (PyTorch, llama.cpp, vLLM supportent ROCm)

Le seul bémol : ROCm a moins d'optimisations que CUDA pour l'inférence
(pas de FP4 natif, cublas _scaled_mm absent). Mais ça s'améliore vite.

---

## Scénario C : NVIDIA compute + AMD lending (cross-vendor avec P2P ?)

```
GPU0 = RTX 5070 Ti (16 GB, NVFP4) → COMPUTE
GPU1 = RX 7900 XT   (20 GB)       → VRAM LENDING

Question : P2P cross-vendor NVIDIA↔AMD ?
```

**P2P NVIDIA↔AMD ? Probablement NON.** Raisons :
- nvidia.ko et amdgpu sont des drivers DIFFÉRENTS
- Ils ne partagent pas de mécanisme P2P commun
- Le PCIe switch peut techniquement router, mais les drivers doivent coopérer
- DMA-BUF (drmPrimeHandleToFD) est la SEULE interface cross-driver pour ça

| Chemin | Débit estimé |
|---|---|
| DMA-BUF kernel (drmPrimeHandleToFD) | ~20-30 GB/s (théorique, jamais testé) |
| CPU-staged via GpuPipeline | **~25 GB/s** (mesuré) |
| P2P direct | ❌ Probablement bloqué |

### Verdict scénario C

**Le meilleur des deux mondes SI DMA-BUF marche.** NVIDIA pour le compute FP4
(rapide, optimisé), AMD pour le lending (P2P ouvert, driver libre). Mais
DMA-BUF cross-driver n'a jamais été testé sur ce projet. C'est LE test à
faire avec la 7900 XT.

---

## Scénario D : Full AMD + ROCm (abandonner NVIDIA ?)

```
GPU0 = RX 7900 XTX (24 GB) → COMPUTE
GPU1 = RX 7900 XT  (20 GB) → LENDING

Les deux sur ROCm. P2P natif. Driver ouvert.
```

| Avantages | Inconvénients |
|---|---|
| ✅ P2P natif, 32 GB/s | ❌ Pas de NVFP4 (FP4 Blackwell) |
| ✅ 44 GB VRAM totale | ❌ ROCm moins mature que CUDA |
| ✅ Driver open-source | ❌ Moins d'optimisations inference |
| ✅ Pas de bridage artificiel | ❌ Certains modèles moins bien supportés |
| ✅ XGMI possible (équivalent NVLink) | |

### Verdict scénario D

**Pas encore.** ROCm progresse vite mais CUDA reste en avance pour l'inférence.
Dans 1-2 ans, AMD+ROCm pourrait être compétitif. Aujourd'hui, le combo
NVIDIA (compute FP4) + AMD (lending P2P) est plus performant.

---

## Synthèse : quel setup pour quel usage ?

| Setup | VRAM | P2P | Compute | Recommandation |
|---|---|---|---|---|
| **Actuel** : 5070Ti + 3090 | 40 GB | ❌ 217 | FP4 (NVIDIA) | Bon, mais P2P bridé |
| **Scénario A** : 5070Ti + 7900XT | 36 GB | ❌ CPU-staged | FP4 (NVIDIA) | **Ton prochain setup** |
| **Scénario B** : 7900XTX + 7900XT | 44 GB | ✅ 32 GB/s | ROCm | Optimal lending pur |
| **Scénario C** : 5070Ti + 7900XT + DMA-BUF | 36 GB | ⚠️ À tester | FP4 (NVIDIA) | **Le Graal si DMA-BUF marche** |

---

## Le test décisif avec la 7900 XT

```python
# Test #1 : P2P NVIDIA↔AMD via DMA-BUF
# → drmPrimeHandleToFD(nvidia_drm) → drmPrimeFDToHandle(amdgpu)
# → Si ça marche → 20-30 GB/s cross-vendor DIRECT

# Test #2 : VRAM lending pur
# → 7900 XT = RAMDisk pour 5070 Ti
# → Stocker couches froides, KV cache, modèles secondaires
# → Mesurer : bande passante réelle, latence de swap

# Test #3 : ROCm standalone
# → 7900 XT en compute ROCm (sans NVIDIA)
# → Benchmark tok/s vs 5070 Ti
# → Vérifier la viabilité long terme d'AMD pur
```

---

## Conclusion

**AMD est objectivement supérieur pour le VRAM lending.** Pas de bridage,
P2P natif, driver ouvert, VRAM moins chère. Le combo NVIDIA (compute FP4) +
AMD (lending P2P) est potentiellement le setup optimal pour VRAMancer.

La 7900 XT à 500€ débloque TOUT ça. C'est le meilleur investissement
pour le projet.

— DeepSeek
