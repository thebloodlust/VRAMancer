# Mesure disagg prefill/décode — RÉFUTÉ sur 2 GPU commodity (sans NVLink)

> `benchmarks/bench_disagg_prefill_decode.py`, Qwen2.5-1.5B fp16, 3090 + 5070 Ti.
> On n'a PAS construit un serveur disagg complet (méthode du probe MoE : mesurer les
> quantités qui décident). Date : 2026-06-14.

## Mesures robustes
| Mesure | Valeur |
|---|---|
| **[A]** prefill / décode (1 stream, prompt 216 tok, 64 gen) | **30 ms / 1736 ms → ratio 58:1** |
| **[B]** taxe transfert KV GPU0→GPU1 (6.2 Mo) | **24 ms, 0.3 GB/s** (latency-bound, P2P bloqué → CPU-staged) |
| **[C]** scaling décode en batch (1 GPU) | B1=36 · B2=73 · **B8=271 tok/s** (quasi-linéaire) ; prefill reste 101 ms à B8 |
| Équilibre prefill≈décode | **~60 streams décode/GPU** |

## Verdict : disagg ne paie pas ici
Le décode **domine le prefill 58:1**. Dédier un GPU au prefill (le principe du disagg)
le laisserait **~2% occupé** — gâché. Et le transfert KV inter-GPU est **latency-bound**
(P2P bloqué en VM/sans NVLink → CPU-staged 0.3 GB/s sur 6 Mo).

Le **vrai levier de débit** mesuré ici : **batcher le décode sur UN GPU** (gratuit,
×7.5 à batch 8). Pour aller plus loin : **data-parallel** (répliquer le modèle, router
les requêtes entières — zéro transfert) quand le modèle tient sur 1 GPU.

Disagg ne deviendrait intéressant que si **prefill ≈ décode** (prompts énormes ET/OU
~60+ streams décode batchés saturant le GPU décode) **ET** avec un **P2P rapide
(NVLink)**. Aucune de ces conditions sur 2 GPU consumer sans NVLink. → Confirme le
caveat 1 (multi-user only) ET ajoute la réalité matérielle (no-NVLink + décode-dominé).

## Notes d'honnêteté (artefacts écartés)
- Une 1re version mesurait le data-parallel par **threads** → ×0.97 : **artefact du GIL**
  (la boucle de décode Python ne parallélise pas). Écarté — un vrai data-parallel exige
  des **process** séparés. Le verdict ne s'appuie PAS dessus.
- Le disagg 1-requête (+19%) inclut ma **boucle décode manuelle** (vs `generate` optimisé),
  pas seulement le transfert. Indicatif, non décisif.
- Verdict fondé uniquement sur **[A] ratio décode/prefill** + **[B] taxe KV** (robustes).

## Implication
5e fois que la mesure corrige l'intuition (A1, GpuPipeline, packing, MoE, **disagg**).
Le pivot « tâches complémentaires » au niveau GPU ne paie pas non plus sur ce matériel.
La valeur reste : **packaging** (S1 drop-in, S2 quickstart) + **optims prouvées** +
**continuous batching** (déjà dans `core/continuous_batcher.py`) — pas un nouveau
moteur de split GPU.
