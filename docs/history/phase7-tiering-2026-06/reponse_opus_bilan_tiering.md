# Opus — bilan perf tiering (5 variantes) + 2 questions avant le MoE

> Suite v0/v0.1/v0.2. Tout mesuré, 1.5B / 4 couches froides, sortie bit-identique.

## Résultats mesurés
| Variante | Correct | VRAM-eff | tok/s |
|---|---|---|---|
| POC un-sens (torch.to) | ✅ | ✅ | 71.7% |
| v0 prefetch per-layer | ✅ | ❌ | 78.3% |
| **v0.1 double-buffer torch** | ✅ | ✅ | **73.1%** (meilleur correct+VRAM-eff) |
| v0.2 **GpuPipeline** | ✅ | ✅ | **61.2%** (plus lent !) |

## Conclusions honnêtes (mesure > intuition)
1. **Mécanisme tiering OK** (correct + VRAM-eff) sur 5 variantes.
2. **Dense = coût ~27%, transfert-bound** ; ne descend pas plus.
3. **GpuPipeline PLUS LENT** : `is_p2p=False` → CPU-stage (2 sauts) + overhead de
   setup PAR APPEL. Je transfère **par paramètre** (petits ~MB sur 1.5B) → l'overhead
   domine. Son 25 GB/s ne vaut que pour de **gros** transferts (256 MB). **On pensait
   tous les deux >90%, la mesure dit 61%.** Bonne leçon.
4. Meilleur dense = **v0.1 torch (73.1%)**.

## Question 1 — batcher les transferts pour réveiller GpuPipeline ?
Idée v0.3 : au lieu de `gp.transfer` PAR PARAM, **pré-packer les params d'une couche
en UN buffer contigu sur cuda:1** (au chargement), faire **UN seul** `gp.transfer`
(gros → la BW 25 GB/s s'exprime), puis re-viewer le dst en params individuels.
Ça amortit l'overhead par-appel. → D'accord que c'est la clé pour que GpuPipeline
gagne ? Ça vaut un v0.3 (ou c'est l'effort de packing qui tue le gain) ?

## Question 2 — hooker le routing MoE (le vrai test de valeur)
Pour le MoE (Qwen3.6-35B-A3B), il faut streamer **seulement les experts activés**.
Mais le routeur (gate) choisit les top-k experts **À L'INTÉRIEUR** du forward de la
couche MoE — donc on ne sait PAS avant. Options que je vois :
- (a) Splitter le forward MoE : gate d'abord (cheap) → prefetch des experts choisis
  → FFN experts. Plus invasif qu'un pre_hook.
- (b) Garder tous les experts sur le store, transférer les sélectionnés après le
  gate (latence gate→fetch par token).
Ton avis sur **le point d'accroche dans Qwen3-MoE** (classe du routeur ? hook sur
`mlp.gate` ?) et la faisabilité avant que je code le test de valeur ?

## État
Tout commité sur `phase7/A1-parity`. Mécanisme prouvé, perf dense honnête (~73%),
GpuPipeline démystifié. Prochain = MoE (où le tiering a sa carte : peu de transfert).

— Opus
