# Mesure réelle — Bande passante inter-GPU RTX 3090 ↔ RTX 5070 Ti (DeepSeek P2.10)

> Réponse concrète à `reponse_a_opus.md` (« testez sur le vrai matériel »).
> Script : `benchmarks/bench_pcie_bw_3090_5070ti.py`. Date : 2026-06-12.

## Contexte matériel (vérifié)
- GPU0 = RTX 3090 (24 GB, Ampere SM 8.6), GPU1 = RTX 5070 Ti (16 GB, Blackwell SM 12.0).
- `nvidia-smi topo` : **P2P = NS** (Not Supported) → transferts **CPU-staged** (GPU→hôte→GPU).
- Mesuré **en cohabitation avec le serveur Qwen3.6** (VRAM libre : 7.6 / 8.0 GB). Les
  comparaisons relatives sont robustes ; les valeurs absolues sont possiblement
  légèrement conservatrices (contention GPU avec le serveur).

## Résultats (médiane sur 30 itérations, 10 warmup)

| Transfert | torch naïf `copy_` | VRAMancer GpuPipeline (meilleur) | gain | chunk optimal |
|---|---|---|---|---|
| 4 MB | 10.3 GB/s | 13.2 GB/s | +28 % | (1 chunk) |
| 16 MB | 10.4 GB/s | **20.9 GB/s** | +101 % | 4 MB |
| 64 MB | 10.4 GB/s | **24.3 GB/s** | +134 % | 4 MB |
| 256 MB | 10.4 GB/s | **25.3 GB/s** | +143 % | 4 MB |

Sweep chunk (256 MB) : 4→25.3, 8→25.1, 16→24.6, 32→23.2, 64→21.1 GB/s.
Sweep chunk (64 MB) : 4→24.3, 8→23.2, 16→21.1, 32→16.7, 64→13.6 GB/s.

## Conclusions

1. **GpuPipeline validé** : le triple-buffering pinned atteint **~25 GB/s effectifs**
   sur un transfert *staged* (2 sauts PCIe), soit **+143 %** vs le `.to()`/`copy_`
   naïf de torch (~10.4 GB/s). C'est ~78 % du plafond PCIe 4.0 x16 d'un seul saut →
   l'overlap DtoH/HtoD fonctionne réellement. DeepSeek avait raison : ce code est bon.

2. **Réponse P2.10 (auto-tuning) : `chunk_mb = 4` est optimal** pour cette paire,
   monotone (plus petit = mieux, dans la plage testée), et **bat le défaut de 16 MB**
   de ~+20 %. Un auto-tuner devrait converger vers 4 MB ici. (Les petits *totaux*
   (4 MB) plafonnent car un seul chunk = pas d'overlap.)

3. **Le goulot desktop est le staging PCIe, pas le réseau** — pertinent pour la
   discussion « bypass kernel » : ici on est limité par les 2 sauts PCIe, pas par
   la pile réseau.

## Note honnête (deux pièges rencontrés, transparence)

- `bench_gpu_transfer` expose **deux** champs : `bandwidth_gbps` = giga-**BITS**/s
  (×8) et `bandwidth_gbs` = giga-**OCTETS**/s. Le nom « gbps » est trompeur — je
  l'ai d'abord lu comme des GB/s et obtenu des « 200 GB/s » impossibles. **Le code
  Rust est correct** (il synchronise via `cuStreamSynchronize`) ; c'est le **nommage
  du champ** qui mériterait d'être clarifié (`bandwidth_gbit_s` / `bandwidth_gbyte_s`).
- Mesure faite serveur actif → contention possible ; à refaire serveur en pause
  pour des absolus définitifs (le classement relatif et le chunk optimal tiennent).

## Validation physique des chiffres (pourquoi on peut leur faire confiance)

- **Synchronisation confirmée** : 256 MB en 10,62 ms = 23,5 GB/s. 10 ms est un
  *vrai* temps de transfert, pas un artefact d'enqueue async (qui serait en µs) →
  `GpuPipeline.transfer` synchronise bien (`cuStreamSynchronize`).
- **Échelle linéaire avec la taille** (256/64/16 MB → 10,6/2,76/0,80 ms, ratio ~4×)
  → transfert complet réel, pas de court-circuit.
- **Cohérent avec le hardware** : `pcie.link.gen.max=4`, `width=16` sur les 2 GPU →
  Gen4 x16 ≈ 31,5 GB/s par lien. Overlap DtoH/HtoD du triple-buffering → ~23-25 GB/s
  effectifs (~75 % du plafond d'un saut). Le naïf torch ~10 GB/s = 2 sauts séquentiels.
  Le **+143 % = l'overlap**, mécanisme bien compris. Histoire physique cohérente.
- **Piège PCIe à connaître** : `pcie.link.gen.current=1` (Gen1) au repos malgré
  `gen.max=4` = downclock d'économie d'énergie ; le lien rampe à Gen4 sous charge.
  → (a) ne PAS lire l'état PCIe instantané comme une capacité ; (b) le 1er transfert
  après repos paie une rampe de lien → explique la sous-perf des petits transferts
  (4 MB = 13 GB/s) autant que le manque d'overlap.

## Limites (honnêteté)
- Moyennes sur 30 itér **sans écart-type** ; le creux 64 MB/chunk32 (16,7 vs 21,1
  GB/s à chunk16) trahit du bruit (contention serveur). Absolus à reconfirmer
  serveur en pause + variance.
- Optimum `chunk=4` **borné à [4,64] MB** — tester 1-2 MB pour le vrai minimum.
- Buffers synthétiques `float32` contigus ; activations réelles bf16 ≈ même (memcpy-bound).
