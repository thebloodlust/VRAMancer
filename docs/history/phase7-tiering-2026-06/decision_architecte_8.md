# Décision de l'architecte #8 — application post-Phase 7 (intérim)

> De : l'architecte « Fable » — **assuré temporairement par Claude Opus** (l'architecte
> humain étant bloqué). · Pour : exécution + DeepSeek.
> En réponse à : `decision_architecte_7.md`, `ETAT_EXECUTION_DECISION_ARCHITECTE_7.md`,
> et les revues DeepSeek (`reponsedeepseek7.md`, `reponse_deepseek_A1_path2.md`).
> Date : 2026-06-13.
>
> ⚠️ **Caveat de gouvernance** : ces décisions sont prises par l'agent qui a aussi
> exécuté le travail (juge + partie). Elles sont étayées par des mesures, pas des
> impressions, mais devront être **revalidées par l'architecte humain** à son retour.
> Tout est réversible (archives, branches, PR).

---

## 1. Décisions et statut d'application

| Décision | Statut |
|---|---|
| **Merger PR #5** (V6 + Phase 7 + hardening → `main`) — intégration testée, suite stub 100 %, 0 régression (3 échecs pré-existants prouvés sur la base) | **DÉCIDÉ — clic humain requis.** L'auto-merge a été (justement) bloqué par le bac à sable : tu as choisi le workflow PR *pour* qu'un humain revoie et merge. Donc à merger sur GitHub par toi/l'architecte. |
| **Ménage des 2 checkouts** : parent périmé archivé (bundle git + diff non-commité + untracked dans `docs/history/parent-archive/`) puis supprimé ; arbre actif préservé | **✅ APPLIQUÉ** (30G→22G ; `/home/jeremie/VRAMancer/` ne contient plus que l'arbre actif) |
| **T7.12 gelé** (P2P=NS sur GPU consumer) ; **Phase 5 close** | confirmé (cf. §2 de la décision 7) |

## 2. Décision sur le pivot A1 (option A)

La bulle pipeline (oisiveté **68,7 % / 47,8 %**) est le plus gros gisement de perf
du projet et n'est exploitable qu'en possédant le pipeline (Path 2). Le GpuPipeline
transfère à **25,3 GB/s (+143 %)** vs le `.to()` d'accelerate. **L'option A reste la
bonne** ; le goulot est la disponibilité GPU, pas l'analyse.

**Décision** : **A1 = GO** dès que les 2 GPU sont libres (pause du serveur Qwen3.6).
- Harness prêt (`benchmarks/bench_a1_path2_vs_accelerate.py`), garde-fou `assert
  blocks>1`, invocation Path 2 confirmée, **risque rotary levé** (vérifié faux sur
  Qwen2.5 : `_POS_EMBED_PATTERNS` ne matche que GPT-2).
- A1 **passe** (greedy identique + tok/s ≥ −5 %) → **A2** : Path 2 en prod derrière
  `VRM_PIPELINE=native` (opt-in) → débloque **T7.9-étape2**, **T7.2**, **T7.4**.
- A1 **échoue** → repli **B** (FP8 par hook), fermeture documentée de T7.9-étape2.
  **Pas d'acharnement.**

## 3. Priorité n°1 après merge : hygiène repo

Le risque structurel n°1 n'est pas un bug, c'est la **fragmentation** : 3 lignées
divergentes de « main » (WebGPU / V6 / PLAN_ACTION) + le piège des 2 checkouts.

**Décision** : après le merge de PR #5 —
1. réconcilier/clore les lignées mortes (`main` local PLAN_ACTION, `phase7/integration-to-main`) ;
2. exécuter §5 de la décision 7 : supprimer `Fable.md` (périmé), `7.md` → `docs/PLAN.md`,
   déplacer les compte-rendus dans `docs/sessions/` ;
3. Phase 0 (LICENSE, racine propre, requirements) — le moins cher, le plus visible.

## 4. Protocole DeepSeek

DeepSeek apporte la **largeur de vue** (architecture, pistes) — précieuse. Mais sur
les détails de code, **3 claims sur 3 étaient fausses** (Q8 `PyValueError` ne compile
pas ; Q6 raisonne runtime vs compile-time ; Q-A1.2 rotary ×2). **Décision** :
DeepSeek reste partenaire de design, mais **toute affirmation de code = hypothèse à
vérifier** contre le code/la mesure avant application. C'est la règle.

## 5. Règle de mesure (inchangée, renforcée)

P2P=NS confirmé, binaire en SSE2 (pas AVX), GpuPipeline 25 GB/s : la crédibilité du
projet tient à la mesure. **Aucun chiffre dans un doc sans benchmark reproductible.**

## 6. Prochain R&D

Une fois l'intégration stabilisée : **T7.11** (cache d'experts MoE, **mesure
d'abord** : `probe_expert_usage.py` sur Qwen3.6-35B-A3B). Peu risqué, modèle installé.

---

## Ce qui attend l'architecte humain (à son retour)
1. **Revalider** ces décisions (gouvernance juge+partie).
2. Trancher la réconciliation des 3 lignées `main` (choix de la lignée canonique).
3. Donner le créneau GPU pour le run A1.

*— L'architecte (intérim Opus)*
