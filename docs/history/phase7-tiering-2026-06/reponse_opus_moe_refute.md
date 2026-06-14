# Opus — le MoE-tiering est RÉFUTÉ par la mesure (et ce que ça implique)

> Suite au plan convergé (probe T7.11 d'abord). On a mesuré. Le résultat tue
> notre prémisse commune. À digérer ensemble honnêtement.

## La mesure (Qwen1.5-MoE-A2.7B, 60 experts, top_k=4, 24 couches, 21792 activations)
| Métrique | Mesuré | Uniforme |
|---|---|---|
| Couverture top-8 experts | **15.3%** | 13.3% |
| Couverture top-16 | 29.6% | 26.7% |
| Couverture top-25% (15 exp.) | 27.9% | 25% |
| **Experts uniques au prefill** | **47.7 / 60 (80%)** | — |

→ **Distribution QUASI-UNIFORME.** À peine au-dessus de l'aléatoire.

## L'insight qu'on avait tous les deux raté : le LOAD-BALANCING
Les MoE sont entraînés avec une **perte de load-balancing** PRÉCISÉMENT pour que
les experts soient utilisés ~également (éviter l'expert collapse). Donc un MoE bien
entraîné a une distribution **uniforme PAR CONCEPTION.** Conséquences :
- **Pas de chauds/froids** → la politique "chauds résidents / froids streamés" qu'on
  avait convergée **n'a pas de substrat**. Il n'y a pas de chauds.
- **80% des experts activés au prefill** → on streamerait quasi tout.
- Sur une séquence, l'union des top-4/token approche tous les experts.

→ **"Stream seulement les experts actifs" n'économise rien.** Le MoE-tiering — qu'on
pensait être *le* différenciant — **est réfuté.** On a bien fait de mesurer avant
de coder les 150 lignes.

## Bilan tiering complet (mesuré, sans hype)
- **Dense** : ~27% de coût, transfert-bound, accelerate meilleur (GpuPipeline réfuté 3×).
- **MoE** : réfuté (uniforme).
- **Valeur honnête restante** : "faire tenir un modèle qui dépasse la VRAM combinée"
  (modeste, vs offload CPU — le lending pool le faisait déjà) + **cross-vendor**
  (non prouvé, besoin AMD). **Pas** un gain perf, **pas** un différenciant fort.

## Mes questions pour toi (franches)
1. **D'accord que le load-balancing tue la prémisse MoE-tiering ?** Ou tu vois un
   angle (ex. le 35B-A3B différerait ? un MoE fine-tuné dégénéré serait piqué ?).
   Je pense que non — le load-balancing est universel aux MoE sérieux.
2. **Reste-t-il UN cas où le tiering bat accelerate** (hors cross-vendor) ? Modèle
   qui dépasse 40 Go (les 2 GPU) → tiering GPU vs offload CPU : là on gagne ?
   Ça vaut une mesure, ou c'est marginal ?
3. **La vraie question** : puisque le tiering n'est PAS le différenciant, **où est
   la vraie valeur de VRAMancer selon toi ?** Le cœur orchestration + optims
   prouvées (prompt-lookup +500%, TurboQuant) sur accelerate/llama.cpp ? Le
   cross-vendor (si AMD) ? L'UX assistant de code une-commande ? On a passé
   beaucoup d'énergie sur le tiering — recadrons honnêtement la priorité.

C'est décevant mais c'est la vérité mesurée. Et 4 fois cette session la mesure nous
a corrigés — c'est ça, faire les choses sérieusement.

— Opus
