# Probe T7.11 — distribution des experts MoE : le MoE-tiering est RÉFUTÉ

> Mesure `benchmarks/probe_expert_usage.py` sur Qwen1.5-MoE-A2.7B (60 experts,
> top_k=4, 24 couches MoE), 3 prompts de code, 80 tokens décode chacun.
> 21 792 activations mesurées. Date : 2026-06-14.

## La mesure
| Métrique | Mesuré | Uniforme |
|---|---|---|
| Couverture top-4 experts | 7.7% | 6.7% |
| Couverture top-8 | 15.3% | 13.3% |
| Couverture top-16 | 29.6% | 26.7% |
| Couverture top-25% (15 experts) | 27.9% | 25% |
| **Experts uniques au prefill** | **47.7 / 60 (80%)** | — |

## Verdict : distribution QUASI-UNIFORME -> MoE-tiering ne gagne pas

La distribution est à peine au-dessus de l'uniforme (top-8 : 15.3% vs 13.3%). **Et
c'est par conception** : les MoE sont entraînés avec une **perte de load-balancing**
pour répartir l'usage des experts ~également (éviter l'expert collapse). Conséquences :

1. **Pas de chauds/froids exploitables** — les experts sont ~équiprobables. La
   politique "chauds résidents / froids streamés" (notre plan convergé) **ne
   s'applique pas** : il n'y a pas de chauds dominants.
2. **Prefill = 80% des experts activés** → on streamerait quasiment tout au prefill.
3. **Décode** : top-4/token, mais sur une séquence l'union approche tous les experts
   (distribution plate).

→ **Le "stream seulement les experts actifs" n'économise quasi rien.** Le MoE-tiering,
qu'on pensait être *le* différenciant du tiering, **est réfuté par la mesure.**

## Implication honnête pour le projet

Le tiering (dense ET MoE) ne fournit **pas** de gain de perf ni de différenciant fort :
- **Dense** : coût ~27% (transfert-bound), accelerate fait mieux.
- **MoE** : réfuté ici — load-balancing → routing uniforme → besoin de ~tous les experts.

**La valeur honnête restante du tiering** :
- **Faire tenir un modèle qui dépasse la VRAM combinée des GPU** (vs offload CPU lent).
  Modeste, mais réel (le lending pool le démontrait déjà).
- **Cross-vendor** (store AMD) — non prouvé, nécessite un GPU AMD.

Ce n'est PAS la carte différenciante qu'on espérait. À refléter dans le README/scope :
le tiering = "faire tenir l'infaisable" (modeste), pas "plus vite" ni "MoE magique".

## La leçon (discipline)
4 fois cette session, la **mesure a corrigé l'intuition** : A1 (forward cassé, pas
le masque/transfert), GpuPipeline (3× plus lent), et maintenant le MoE (uniforme,
pas piqué). Mesurer d'abord a évité de coder ~150 lignes de streaming d'experts
inutile. C'est exactement la valeur de la méthode — même quand le résultat déçoit.

## Nuance (à vérifier si on veut creuser)
- Mesuré sur Qwen1.5-MoE-A2.7B. Le 35B-A3B pourrait différer légèrement, mais le
  load-balancing est universel aux MoE bien entraînés → peu d'espoir d'une vraie pique.
- Un MoE NON load-balanced (rare) ou un fine-tune dégénéré pourrait être piqué.
